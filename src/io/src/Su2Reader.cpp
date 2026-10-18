#include "Su2Reader.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fvm::io
{
namespace
{
using fvm::core::Index;
using fvm::core::MeshInput;
using fvm::core::PatchInput;
using fvm::core::UnstructuredMesh;

[[noreturn]] void fail(const std::string& filename,
    std::size_t line,
    const std::string& message)
{
    throw std::invalid_argument("Su2Reader: '" + filename + "': line "
        + std::to_string(line) + ": " + message);
}

std::string trim(const std::string& text)
{
    const auto isSpace = [](unsigned char character)
    {
        return std::isspace(character) != 0;
    };

    auto first = text.find_first_not_of(" \t\r\n\f\v");
    if (first == std::string::npos)
    {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\r\n\f\v");
    return text.substr(first, last - first + 1);
}

std::vector<std::string> tokenize(const std::string& line)
{
    std::istringstream stream(line);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token)
    {
        tokens.push_back(token);
    }
    return tokens;
}

bool parseUnsigned(const std::string& token, std::size_t& value)
{
    if (token.empty()
        || !std::all_of(token.begin(), token.end(), [](unsigned char character)
        {
            return std::isdigit(character) != 0;
        }))
    {
        return false;
    }

    try
    {
        std::size_t position = 0;
        const unsigned long long parsed = std::stoull(token, &position);
        if (position != token.size() || parsed > std::numeric_limits<Index>::max())
        {
            return false;
        }
        value = static_cast<Index>(parsed);
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool parseScalar(const std::string& token, fvm::core::Scalar& value)
{
    if (token.empty())
    {
        return false;
    }

    try
    {
        std::size_t position = 0;
        const double parsed = std::stod(token, &position);
        if (position != token.size())
        {
            return false;
        }
        value = parsed;
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

std::size_t parseSectionCount(const std::vector<std::string>& tokens,
    const std::string& keyword,
    const std::string& filename,
    std::size_t line)
{
    const std::string prefix = keyword + "=";
    if (tokens.size() < 2 || tokens[0] != prefix)
    {
        fail(filename, line, "expected section '" + prefix + "'");
    }
    if (tokens.size() != 2)
    {
        fail(filename, line, keyword + "= accepts exactly one value");
    }

    std::size_t value = 0;
    if (!parseUnsigned(tokens[1], value))
    {
        fail(filename, line, keyword + "= value must be a non-negative integer");
    }
    return value;
}

bool getNextNonBlankLine(std::istream& stream, std::string& line, std::size_t& lineNumber)
{
    while (std::getline(stream, line))
    {
        ++lineNumber;
        if (!trim(line).empty())
        {
            return true;
        }
    }
    return false;
}

bool getNextDataLine(std::istream& stream, std::string& line, std::size_t& lineNumber)
{
    ++lineNumber;
    return static_cast<bool>(std::getline(stream, line));
}

Index pointIndex(const std::unordered_map<Index, Index>& pointIds,
    const std::string& token,
    const std::string& filename,
    std::size_t line)
{
    Index id = 0;
    if (!parseUnsigned(token, id)
        || pointIds.find(id) == pointIds.end())
    {
        fail(filename, line, "unknown node id '" + token + "'");
    }
    return pointIds.at(id);
}

void readPoints(std::istream& stream,
    std::size_t count,
    MeshInput& input,
    std::unordered_map<Index, Index>& pointIds,
    const std::string& filename,
    std::size_t& line)
{
    for (std::size_t local = 0; local < count; ++local)
    {
        std::string text;
        if (!getNextDataLine(stream, text, line))
        {
            fail(filename, line, "NPOIN section has fewer than " + std::to_string(count)
                + " point lines");
        }

        const std::vector<std::string> tokens = tokenize(text);
        if (tokens.size() != 3)
        {
            fail(filename, line, "a point line must contain exactly id x y");
        }

        Index id = 0;
        if (!parseUnsigned(tokens[0], id) || id == 0)
        {
            fail(filename, line, "point id '" + tokens[0] + "' must be a positive integer");
        }
        if (!pointIds.emplace(id, input.points.size()).second)
        {
            fail(filename, line, "duplicate point id " + std::to_string(id));
        }

        fvm::core::Point2 point;
        if (!parseScalar(tokens[1], point.x) || !parseScalar(tokens[2], point.y))
        {
            fail(filename, line, "point coordinates must be numeric");
        }
        input.points.push_back(point);
    }
}

std::size_t nodeCountForType(std::size_t type)
{
    if (type == 5)
    {
        return 3;
    }
    if (type == 9)
    {
        return 4;
    }
    return 0;
}

void readElements(std::istream& stream,
    std::size_t count,
    MeshInput& input,
    const std::unordered_map<Index, Index>& pointIds,
    const std::string& filename,
    std::size_t& line)
{
    for (std::size_t local = 0; local < count; ++local)
    {
        std::string text;
        if (!getNextDataLine(stream, text, line))
        {
            fail(filename, line, "NELEM section has fewer than " + std::to_string(count)
                + " element lines");
        }

        const std::vector<std::string> tokens = tokenize(text);
        if (tokens.empty())
        {
            fail(filename, line, "empty element line");
        }

        std::size_t type = 0;
        if (!parseUnsigned(tokens[0], type))
        {
            fail(filename, line, "invalid element type '" + tokens[0] + "'");
        }
        const std::size_t nodeCount = nodeCountForType(type);
        if (nodeCount == 0)
        {
            fail(filename, line, "unsupported element type " + std::to_string(type));
        }

        const std::size_t minimumSize = nodeCount + 1;
        const std::size_t maximumSize = nodeCount + 2;
        if (tokens.size() < minimumSize || tokens.size() > maximumSize)
        {
            fail(filename, line, "element type " + std::to_string(type)
                + " must contain " + std::to_string(nodeCount)
                + " node ids and an optional element id");
        }

        std::vector<Index> loop;
        loop.reserve(nodeCount);
        for (std::size_t node = 1; node <= nodeCount; ++node)
        {
            loop.push_back(pointIndex(pointIds, tokens[node], filename, line));
        }

        if (tokens.size() == maximumSize)
        {
            Index elementId = 0;
            if (!parseUnsigned(tokens.back(), elementId))
            {
                fail(filename, line, "trailing element id '" + tokens.back()
                    + "' must be an integer");
            }
        }
        input.cells.push_back(std::move(loop));
    }
}

void readMarkers(std::istream& stream,
    std::size_t count,
    MeshInput& input,
    const std::unordered_map<Index, Index>& pointIds,
    const std::string& filename,
    std::size_t& line)
{
    std::set<std::string> names;
    for (std::size_t marker = 0; marker < count; ++marker)
    {
        std::string text;
        if (!getNextNonBlankLine(stream, text, line))
        {
            fail(filename, line, "NMARK section has fewer than " + std::to_string(count)
                + " markers");
        }

        const std::string tagPrefix = "MARKER_TAG=";
        if (text.compare(0, tagPrefix.size(), tagPrefix) != 0)
        {
            fail(filename, line, "expected marker section 'MARKER_TAG='");
        }

        PatchInput patch;
        patch.name = trim(text.substr(tagPrefix.size()));
        if (patch.name.empty())
        {
            fail(filename, line, "marker name must not be empty");
        }
        if (!names.insert(patch.name).second)
        {
            fail(filename, line, "duplicate marker name '" + patch.name + "'");
        }

        std::string elemsText;
        if (!getNextNonBlankLine(stream, elemsText, line))
        {
            fail(filename, line, "marker '" + patch.name + "' has no MARKER_ELEMS= line");
        }
        const std::size_t edgeCount = parseSectionCount(tokenize(elemsText),
            "MARKER_ELEMS",
            filename,
            line);

        patch.edges.reserve(edgeCount);
        for (std::size_t local = 0; local < edgeCount; ++local)
        {
            std::string edgeText;
            if (!getNextDataLine(stream, edgeText, line))
            {
                fail(filename, line, "marker '" + patch.name + "' has fewer than "
                    + std::to_string(edgeCount) + " element lines");
            }

            const std::vector<std::string> tokens = tokenize(edgeText);
            if (tokens.size() != 3)
            {
                fail(filename, line, "a marker element line must contain exactly 3 v1 v2");
            }

            std::size_t type = 0;
            if (!parseUnsigned(tokens[0], type) || type != 3)
            {
                fail(filename, line, "marker '" + patch.name
                    + "' supports only line element type 3");
            }

            const Index first = pointIndex(pointIds, tokens[1], filename, line);
            const Index second = pointIndex(pointIds, tokens[2], filename, line);
            patch.edges.emplace_back(first, second);
        }
        input.patches.push_back(std::move(patch));
    }
}

void validateMarkerEdges(const MeshInput& input, const std::string& filename)
{
    std::map<std::pair<Index, Index>, Index> edgeCounts;
    for (const std::vector<Index>& cell : input.cells)
    {
        for (std::size_t local = 0; local < cell.size(); ++local)
        {
            const Index first = cell[local];
            const Index second = cell[(local + 1) % cell.size()];
            ++edgeCounts[std::minmax(first, second)];
        }
    }

    for (const PatchInput& patch : input.patches)
    {
        for (const auto& [first, second] : patch.edges)
        {
            const auto key = std::minmax(first, second);
            const auto edge = edgeCounts.find(key);
            const Index occurrence = edge == edgeCounts.end() ? 0 : edge->second;
            if (occurrence == 0)
            {
                fail(filename, 0, "marker '" + patch.name
                    + "' contains edge (" + std::to_string(first) + ", "
                    + std::to_string(second) + ") that is not a cell edge");
            }
            if (occurrence == 2)
            {
                fail(filename, 0, "marker '" + patch.name
                    + "' contains an interior edge");
            }
            if (occurrence != 1)
            {
                fail(filename, 0, "marker '" + patch.name + "' contains non-manifold edge ("
                    + std::to_string(first) + ", " + std::to_string(second) + ")");
            }
        }
    }
}
UnstructuredMesh buildMesh(MeshInput input, const std::string& filename)
{
    try
    {
        return UnstructuredMesh(input);
    }
    catch (const std::exception& error)
    {
        throw std::invalid_argument("Su2Reader: '" + filename + "': "
            + error.what());
    }
}
} // namespace

UnstructuredMesh Su2Reader::read(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Su2Reader: cannot open '" + filename + "'");
    }

    try
    {
        MeshInput input;
        std::unordered_map<Index, Index> pointIds;
        std::size_t line = 0;

        std::string text;
        if (!getNextNonBlankLine(file, text, line))
        {
            fail(filename, line, "missing section 'NDIME='");
        }
        const std::size_t dimension = parseSectionCount(tokenize(text),
            "NDIME",
            filename,
            line);
        if (dimension != 2)
        {
            fail(filename, line, "NDIME must be 2");
        }

        if (!getNextNonBlankLine(file, text, line))
        {
            fail(filename, line, "missing section 'NPOIN='");
        }
        const std::size_t pointCount = parseSectionCount(tokenize(text),
            "NPOIN",
            filename,
            line);
        if (pointCount == 0)
        {
            fail(filename, line, "NPOIN must be greater than zero");
        }
        readPoints(file, pointCount, input, pointIds, filename, line);

        if (!getNextNonBlankLine(file, text, line))
        {
            fail(filename, line, "missing section 'NELEM='");
        }
        const std::size_t elementCount = parseSectionCount(tokenize(text),
            "NELEM",
            filename,
            line);
        if (elementCount == 0)
        {
            fail(filename, line, "no cells");
        }
        readElements(file, elementCount, input, pointIds, filename, line);

        if (!getNextNonBlankLine(file, text, line))
        {
            fail(filename, line, "missing section 'NMARK='");
        }
        const std::size_t markerCount = parseSectionCount(tokenize(text),
            "NMARK",
            filename,
            line);
        readMarkers(file, markerCount, input, pointIds, filename, line);
        validateMarkerEdges(input, filename);

        return buildMesh(std::move(input), filename);
    }
    catch (const std::invalid_argument&)
    {
        throw;
    }
    catch (const std::exception& error)
    {
        throw std::invalid_argument("Su2Reader: '" + filename + "': " + error.what());
    }
}
} // namespace fvm::io
