#pragma once

#include "Types.h"

using fvm::core::Scalar;

namespace fvm::numerical
{

/**
 * @brief Implicit time-integration scheme for the theta family.
 *
 * The spatial operator is split as theta * L(phi^{n+1}) +
 * (1 - theta) * L(phi^n); see assembleTransientTransport.
 */
enum class TimeScheme
{
    Euler,        ///< Backward Euler, theta = 1 (first order).
    CrankNicolson ///< theta = 1/2 (second order).
};

/**
 * @brief theta weight of the new time level for a TimeScheme.
 */
inline Scalar thetaOf(TimeScheme scheme)
{
    switch (scheme)
    {
        case TimeScheme::CrankNicolson:
            return 0.5;
        case TimeScheme::Euler:
        default:
            return 1.0;
    }
}

/**
 * @brief Transient ddt term for the theta scheme.
 *
 * `dt <= 0` means steady state: the ddt term is omitted and the assembly
 * reduces to the steady operators. For incompressible momentum the
 * transported quantity is rho * u, hence the density factor; scalar
 * transport does not use this struct (its ddt coefficient is V/dt).
 */
struct TimeTerm
{
    Scalar rho = 1.0;               ///< Density (momentum only).
    Scalar dt = 0.0;                ///< Time-step size; <= 0 means steady.
    TimeScheme scheme = TimeScheme::Euler;

    bool active() const
    {
        return dt > 0.0;
    }
    Scalar theta() const
    {
        return thetaOf(scheme);
    }
};

} // namespace fvm::numerical
