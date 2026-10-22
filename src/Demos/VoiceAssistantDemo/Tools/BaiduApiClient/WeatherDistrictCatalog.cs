using System.Reflection;
using System.Text;

namespace VoiceAssistantDemo.Tools.BaiduApiClient;

internal sealed class WeatherDistrictCatalog
{
    private const string ResourceName =
        "VoiceAssistantDemo.Resources.weather_district_id.csv";
    private static readonly Lazy<WeatherDistrictCatalog> SharedCatalog = new(
        LoadEmbedded,
        LazyThreadSafetyMode.ExecutionAndPublication);

    private readonly DistrictRecord[] _districts;

    private WeatherDistrictCatalog(DistrictRecord[] districts)
        => _districts = districts;

    public static WeatherDistrictCatalog Shared => SharedCatalog.Value;

    public DistrictRecord Resolve(string name, string? province, string? city)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        string target = name.Trim();

        List<DistrictRecord> matches = MatchByAdministrativeName(
            _districts,
            static row => row.District,
            target);
        if (matches.Count == 0)
        {
            matches = MatchByAdministrativeName(
                _districts,
                static row => row.City,
                target);
        }
        if (matches.Count == 0)
        {
            matches = _districts
                .Where(row => MatchesQualifiedName(row, target))
                .ToList();
        }

        if (!string.IsNullOrWhiteSpace(province))
        {
            matches = matches
                .Where(row => SameAdministrativeName(row.Province, province))
                .ToList();
        }
        if (!string.IsNullOrWhiteSpace(city))
        {
            matches = matches
                .Where(row => SameAdministrativeName(row.City, city))
                .ToList();
        }

        matches = matches.DistinctBy(static row => row.DistrictId).ToList();
        if (matches.Count == 1)
            return matches[0];
        if (matches.Count == 0)
        {
            throw new WeatherLocationException(
                $"行政区划表中找不到“{target}”，请换用城市、区县名称或行政区划 ID。");
        }

        string candidates = string.Join(
            "；",
            matches.Take(8).Select(static row =>
                $"{row.Province}/{row.City}/{row.District}({row.DistrictId})"));
        throw new WeatherLocationException(
            $"“{target}”对应多个位置：{candidates}。请补充省份或城市。");
    }

    private static List<DistrictRecord> MatchByAdministrativeName(
        IEnumerable<DistrictRecord> rows,
        Func<DistrictRecord, string> selector,
        string target)
    {
        List<DistrictRecord> exact = rows
            .Where(row => string.Equals(
                selector(row).Trim(),
                target,
                StringComparison.OrdinalIgnoreCase))
            .ToList();
        return exact.Count > 0
            ? exact
            : rows.Where(row => SameAdministrativeName(selector(row), target)).ToList();
    }

    private static bool MatchesQualifiedName(DistrictRecord row, string target)
    {
        string compact = RemoveWhitespace(target);
        string district = NormalizeName(row.District);
        if (!compact.EndsWith(district, StringComparison.OrdinalIgnoreCase))
            return false;

        string province = NormalizeName(row.Province);
        string city = NormalizeName(row.City);
        return compact.Contains(province, StringComparison.OrdinalIgnoreCase) ||
            compact.Contains(city, StringComparison.OrdinalIgnoreCase);
    }

    private static bool SameAdministrativeName(string left, string right) =>
        string.Equals(
            NormalizeName(left),
            NormalizeName(right),
            StringComparison.OrdinalIgnoreCase);

    private static string NormalizeName(string value)
    {
        string result = RemoveWhitespace(value);
        string[] suffixes =
        [
            "特别行政区",
            "维吾尔自治区",
            "壮族自治区",
            "回族自治区",
            "自治区",
            "自治州",
            "地区",
            "省",
            "市",
            "区",
            "县",
            "旗",
            "盟",
        ];
        foreach (string suffix in suffixes)
        {
            if (result.EndsWith(suffix, StringComparison.Ordinal) &&
                result.Length > suffix.Length)
            {
                return result[..^suffix.Length];
            }
        }
        return result;
    }

    private static string RemoveWhitespace(string value) =>
        string.Concat(value.Where(static character => !char.IsWhiteSpace(character)));

    private static WeatherDistrictCatalog LoadEmbedded()
    {
        Assembly assembly = typeof(WeatherDistrictCatalog).Assembly;
        using Stream stream = assembly.GetManifestResourceStream(ResourceName)
            ?? throw new InvalidOperationException(
                $"Embedded weather district resource '{ResourceName}' was not found.");
        using var reader = new StreamReader(
            stream,
            Encoding.UTF8,
            detectEncodingFromByteOrderMarks: true);

        string? header = reader.ReadLine();
        if (!string.Equals(
            header,
            "district_id,province,city,city_geocode,district,district_geocode,lon,lat",
            StringComparison.Ordinal))
        {
            throw new InvalidDataException("Weather district resource has an unexpected header.");
        }

        var rows = new List<DistrictRecord>();
        int lineNumber = 1;
        while (reader.ReadLine() is { } line)
        {
            lineNumber++;
            string[] columns = line.Split(',');
            if (columns.Length != 8)
            {
                throw new InvalidDataException(
                    $"Weather district resource line {lineNumber} has {columns.Length} columns.");
            }
            rows.Add(new DistrictRecord(
                columns[0],
                columns[1],
                columns[2],
                columns[4],
                columns[6],
                columns[7]));
        }

        return rows.Count > 0
            ? new WeatherDistrictCatalog(rows.ToArray())
            : throw new InvalidDataException("Weather district resource is empty.");
    }
}

internal sealed record DistrictRecord(
    string DistrictId,
    string Province,
    string City,
    string District,
    string Longitude,
    string Latitude);

internal sealed class WeatherLocationException(string message) : Exception(message);
