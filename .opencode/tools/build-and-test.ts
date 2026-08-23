import { tool } from "@opencode-ai/plugin"
import path from "path"

const MAX_OUTPUT_CHARS = 8000

function tail(text: string): string {
  if (text.length <= MAX_OUTPUT_CHARS) return text
  return `... (truncated, showing last ${MAX_OUTPUT_CHARS} chars)\n` + text.slice(-MAX_OUTPUT_CHARS)
}

function run(cmd: string[], cwd: string): { exitCode: number; output: string } {
  const result = Bun.spawnSync({
    cmd,
    cwd,
    stdout: "pipe",
    stderr: "pipe",
  })
  const stdout = result.stdout.toString()
  const stderr = result.stderr.toString()
  return { exitCode: result.exitCode, output: (stdout + (stderr ? "\n" + stderr : "")).trim() }
}

export default tool({
  description:
    "Configure (cmake -G Ninja) and build this FVM-solver project in Debug or Release mode (default Release), " +
    "building fvm_solver and/or fvm_tests (default both), then optionally run ctest and/or fvm_solver. " +
    "Use this instead of invoking cmake directly for project builds. " +
    "IMPORTANT: when the user directly requests a build/test, confirm the options (config, target, run) " +
    "via the question tool first, unless the user already stated them explicitly. " +
    "When building as part of a code modification workflow, proceed directly with the defaults " +
    "(config=Release, target=all, run=tests) without asking.",
  args: {
    config: tool.schema
      .enum(["Debug", "Release"])
      .default("Release")
      .describe("CMake build type: Debug or Release (default Release)"),
    target: tool.schema
      .enum(["all", "fvm_solver", "fvm_tests"])
      .default("all")
      .describe("Which target(s) to build: all (default), fvm_solver, or fvm_tests"),
    run: tool.schema
      .enum(["none", "tests", "solver", "both"])
      .default("tests")
      .describe(
        "What to run after a successful build: none, tests (ctest, requires fvm_tests), " +
        "solver (run fvm_solver demo, requires fvm_solver), or both (default tests)",
      ),
  },
  async execute(args, context) {
    const cwd = context.worktree ?? context.directory ?? process.cwd()
    const config = args.config ?? "Release"
    const target = args.target ?? "all"
    const runOption = args.run ?? "tests"
    const buildDir = path.join(cwd, "build", config)
    const sections: string[] = []
    let failed = false

    // --- Configure ---
    const configureCmd = [
      "cmake",
      "-S", cwd,
      "-B", buildDir,
      "-G", "Ninja",
      `-DCMAKE_BUILD_TYPE=${config}`,
    ]
    const vcpkgRoot = process.env.VCPKG_ROOT
    if (vcpkgRoot) {
      configureCmd.push(`-DCMAKE_TOOLCHAIN_FILE=${path.join(vcpkgRoot, "scripts", "buildsystems", "vcpkg.cmake")}`)
    }
    sections.push(`$ ${configureCmd.join(" ")}`)
    const configure = run(configureCmd, cwd)
    sections.push(tail(configure.output))
    if (configure.exitCode !== 0) {
      sections.push(`\nCONFIGURE FAILED (exit code ${configure.exitCode})`)
      return sections.join("\n")
    }

    // --- Build ---
    const buildCmd = ["cmake", "--build", buildDir, "--parallel"]
    if (target !== "all") {
      buildCmd.push("--target", target)
    }
    sections.push(`\n$ ${buildCmd.join(" ")}`)
    const build = run(buildCmd, cwd)
    sections.push(tail(build.output))
    if (build.exitCode !== 0) {
      sections.push(`\nBUILD FAILED (exit code ${build.exitCode})`)
      return sections.join("\n")
    }
    sections.push(`\nBUILD SUCCEEDED (${config}, target: ${target})`)

    // --- Run ---
    const runTestsRequested = runOption === "tests" || runOption === "both"
    const runSolverRequested = runOption === "solver" || runOption === "both"

    if (runTestsRequested) {
      if (target === "fvm_solver") {
        sections.push(`\nSKIPPED tests: fvm_tests was not built (target=${target})`)
      } else {
        const testCmd = ["ctest", "--test-dir", buildDir, "--output-on-failure", "-C", config]
        sections.push(`\n$ ${testCmd.join(" ")}`)
        const test = run(testCmd, cwd)
        sections.push(tail(test.output))
        if (test.exitCode !== 0) {
          failed = true
          sections.push(`\nTESTS FAILED (exit code ${test.exitCode})`)
        } else {
          sections.push(`\nALL TESTS PASSED`)
        }
      }
    }

    if (runSolverRequested) {
      if (target === "fvm_tests") {
        sections.push(`\nSKIPPED solver: fvm_solver was not built (target=${target})`)
      } else {
        const exeExt = process.platform === "win32" ? ".exe" : ""
        const solverExe = path.join(buildDir, `fvm_solver${exeExt}`)
        sections.push(`\n$ ${solverExe} (cwd: ${buildDir})`)
        const solver = run([solverExe], buildDir)
        sections.push(tail(solver.output))
        if (solver.exitCode !== 0) {
          failed = true
          sections.push(`\nSOLVER FAILED (exit code ${solver.exitCode})`)
        } else {
          sections.push(`\nSOLVER RUN SUCCEEDED`)
        }
      }
    }

    const summary = failed ? "RESULT: FAILED" : "RESULT: SUCCESS"
    sections.push(`\n${summary}`)
    return sections.join("\n")
  },
})
