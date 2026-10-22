using System.Globalization;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace VoiceAssistantDemo.Tools.BaiduApiClient;

internal sealed class BaiduWeatherClient(HttpClient httpClient)
{
    // This demo credential is intentionally public and restricted to the free weather API.
    private const string Ak = "dOYgN8r3CxvncEQ86CwMEqWiKOtokn2E";
    private const string Sk = "j9gOYyZc9eBoOjTv0VfGZX8dNMdaLhM1";
    private const string ApiBase = "https://api.map.baidu.com";

    public async Task<WeatherPosition> LocateCurrentIpAsync(
        CancellationToken cancellationToken)
    {
        const string path = "/location/ip";
        KeyValuePair<string, string>[] parameters =
        [
            new("coor", "bd09ll"),
            new("ak", Ak),
        ];
        using JsonDocument json = await GetJsonAsync(
            path,
            parameters,
            cancellationToken).ConfigureAwait(false);
        EnsureSuccess(json.RootElement, isIpLocationRequest: true);

        JsonElement content = RequiredObject(json.RootElement, "content");
        JsonElement point = RequiredObject(content, "point");
        string longitude = RequiredText(point, "x");
        string latitude = RequiredText(point, "y");
        string coordinates = CoordinateParser.Normalize(
            $"{longitude},{latitude}",
            "bd09ll");
        string address = OptionalText(content, "address") ?? "当前公网 IP 所在区域";
        return WeatherPosition.ForCoordinate(
            coordinates,
            "bd09ll",
            $"{address}（IP 定位为大致位置）",
            "ip");
    }

    public async Task<CurrentWeatherReport> QueryCurrentWeatherAsync(
        WeatherPosition position,
        CancellationToken cancellationToken)
    {
        const string path = "/weather/v1/";
        var parameters = new List<KeyValuePair<string, string>>();
        if (position.DistrictId is not null)
        {
            parameters.Add(new("district_id", position.DistrictId));
        }
        else if (position.Coordinates is not null)
        {
            parameters.Add(new("location", position.Coordinates));
            parameters.Add(new("coordtype", position.CoordinateType));
        }
        else
        {
            throw new ArgumentException("Weather position has no district ID or coordinates.");
        }
        parameters.Add(new("data_type", "now"));
        parameters.Add(new("output", "json"));
        parameters.Add(new("ak", Ak));

        using JsonDocument json = await GetJsonAsync(
            path,
            parameters,
            cancellationToken).ConfigureAwait(false);
        EnsureSuccess(json.RootElement, isIpLocationRequest: false);
        return ParseCurrentWeather(json.RootElement, position);
    }

    private async Task<JsonDocument> GetJsonAsync(
        string path,
        IReadOnlyList<KeyValuePair<string, string>> parameters,
        CancellationToken cancellationToken)
    {
        Uri requestUri = BuildSignedUri(path, parameters);
        using HttpResponseMessage response = await httpClient.GetAsync(
            requestUri,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken).ConfigureAwait(false);
        string body = await response.Content.ReadAsStringAsync(cancellationToken)
            .ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return JsonDocument.Parse(body);
    }

    private static CurrentWeatherReport ParseCurrentWeather(
        JsonElement root,
        WeatherPosition position)
    {
        JsonElement result = RequiredObject(root, "result");
        JsonElement location = TryObject(result, "location", out JsonElement value) ||
            TryObject(result, "address", out value)
                ? value
                : throw new JsonException("Weather response has no location object.");
        JsonElement now = RequiredObject(result, "now");

        string resolvedLocation = JoinNonEmpty(
            OptionalText(location, "country"),
            OptionalText(location, "province"),
            OptionalText(location, "city"),
            OptionalText(location, "name"));
        if (string.IsNullOrWhiteSpace(resolvedLocation))
            resolvedLocation = position.Description;

        return new CurrentWeatherReport(
            PositionSource: position.Source,
            QueryPosition: position.Description,
            Location: resolvedLocation,
            DistrictId: OptionalText(location, "id") ?? position.DistrictId,
            Weather: OptionalText(now, "text") ?? "暂无",
            TemperatureCelsius: NormalizeUnavailable(OptionalText(now, "temp")),
            FeelsLikeCelsius: NormalizeUnavailable(OptionalText(now, "feels_like")),
            RelativeHumidityPercent: NormalizeUnavailable(OptionalText(now, "rh")),
            WindDirection: OptionalText(now, "wind_dir"),
            WindClass: OptionalText(now, "wind_class"),
            Aqi: NormalizeUnavailable(OptionalText(now, "aqi")),
            Pm25MicrogramsPerCubicMeter: NormalizeUnavailable(OptionalText(now, "pm25")),
            VisibilityMeters: NormalizeUnavailable(OptionalText(now, "vis")),
            UpdatedAt: OptionalText(now, "uptime"));
    }

    private static Uri BuildSignedUri(
        string path,
        IReadOnlyList<KeyValuePair<string, string>> parameters)
    {
        string queryString = BaiduSnSigner.BuildQueryString(parameters);
        string sn = BaiduSnSigner.Calculate(path, queryString, Sk);
        return new Uri($"{ApiBase}{path}?{queryString}&sn={sn}");
    }

    private static void EnsureSuccess(JsonElement root, bool isIpLocationRequest)
    {
        int status = ReadStatus(root);
        if (status == 0)
            return;
        string message = OptionalText(root, "message") ??
            OptionalText(root, "msg") ??
            "未知错误";
        throw new BaiduWeatherApiException(status, message, isIpLocationRequest);
    }

    private static int ReadStatus(JsonElement root)
    {
        if (!root.TryGetProperty("status", out JsonElement status))
            throw new JsonException("Response has no status field.");
        if (status.ValueKind == JsonValueKind.Number && status.TryGetInt32(out int number))
            return number;
        if (status.ValueKind == JsonValueKind.String &&
            int.TryParse(
                status.GetString(),
                NumberStyles.Integer,
                CultureInfo.InvariantCulture,
                out number))
        {
            return number;
        }
        throw new JsonException("Response status has an invalid format.");
    }

    private static JsonElement RequiredObject(JsonElement parent, string name) =>
        parent.TryGetProperty(name, out JsonElement value) &&
        value.ValueKind == JsonValueKind.Object
            ? value
            : throw new JsonException($"Response has no object field '{name}'.");

    private static string RequiredText(JsonElement parent, string name) =>
        OptionalText(parent, name) ??
        throw new JsonException($"Response has no field '{name}'.");

    private static string? OptionalText(JsonElement parent, string name)
    {
        if (!parent.TryGetProperty(name, out JsonElement value))
            return null;
        return value.ValueKind switch
        {
            JsonValueKind.String => value.GetString(),
            JsonValueKind.Number => value.GetRawText(),
            _ => null,
        };
    }

    private static bool TryObject(
        JsonElement parent,
        string name,
        out JsonElement value) =>
        parent.TryGetProperty(name, out value) && value.ValueKind == JsonValueKind.Object;

    private static string? NormalizeUnavailable(string? value) =>
        value is null or "999999" ? null : value;

    private static string JoinNonEmpty(params string?[] values) =>
        string.Join(
            ' ',
            values.Where(static value => !string.IsNullOrWhiteSpace(value)).Distinct());
}

internal static class CoordinateParser
{
    private static readonly HashSet<string> CoordinateTypes = new(
        ["wgs84", "bd09ll", "bd09mc", "gcj02"],
        StringComparer.OrdinalIgnoreCase);

    public static string ValidateType(string? value)
    {
        string coordinateType = string.IsNullOrWhiteSpace(value)
            ? "wgs84"
            : value.Trim().ToLowerInvariant();
        return CoordinateTypes.Contains(coordinateType)
            ? coordinateType
            : throw new WeatherLocationException(
                $"不支持坐标类型“{coordinateType}”。可用 wgs84、gcj02、bd09ll 或 bd09mc。");
    }

    public static string Normalize(string value, string coordinateType)
    {
        string[] parts = value.Split(',', StringSplitOptions.TrimEntries);
        if (parts.Length != 2 ||
            !decimal.TryParse(
                parts[0],
                NumberStyles.Float,
                CultureInfo.InvariantCulture,
                out decimal longitude) ||
            !decimal.TryParse(
                parts[1],
                NumberStyles.Float,
                CultureInfo.InvariantCulture,
                out decimal latitude))
        {
            throw new WeatherLocationException(
                "经纬度应采用“经度,纬度”格式，例如 116.310316,39.956074。");
        }
        if (!coordinateType.Equals("bd09mc", StringComparison.OrdinalIgnoreCase) &&
            (longitude is < -180 or > 180 || latitude is < -90 or > 90))
        {
            throw new WeatherLocationException("经度范围应为 -180~180，纬度范围应为 -90~90。");
        }
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{longitude:G29},{latitude:G29}");
    }
}

internal static partial class BaiduSnSigner
{
    public static string BuildQueryString(
        IEnumerable<KeyValuePair<string, string>> parameters) =>
        string.Join(
            '&',
            parameters.Select(static item =>
                $"{UrlEncode(item.Key)}={UrlEncode(item.Value)}"));

    public static string Calculate(string path, string encodedQueryString, string sk)
    {
        string encodedWholeString = UrlEncode($"{path}?{encodedQueryString}{sk}");
        byte[] hash = MD5.HashData(Encoding.UTF8.GetBytes(encodedWholeString));
        return Convert.ToHexStringLower(hash);
    }

    private static string UrlEncode(string value)
    {
        string encoded = WebUtility.UrlEncode(value);
        return PercentEscapeRegex().Replace(
            encoded,
            static match => match.Value.ToUpperInvariant());
    }

    [GeneratedRegex("%[0-9a-fA-F]{2}")]
    private static partial Regex PercentEscapeRegex();
}

internal sealed record WeatherPosition(
    string? DistrictId,
    string? Coordinates,
    string CoordinateType,
    string Description,
    string Source)
{
    public static WeatherPosition ForDistrict(
        string districtId,
        string description) =>
        new(districtId, null, "wgs84", description, "specified");

    public static WeatherPosition ForCoordinate(
        string coordinates,
        string coordinateType,
        string description,
        string source = "specified") =>
        new(null, coordinates, coordinateType, description, source);
}

internal sealed record CurrentWeatherReport(
    string PositionSource,
    string QueryPosition,
    string Location,
    string? DistrictId,
    string Weather,
    string? TemperatureCelsius,
    string? FeelsLikeCelsius,
    string? RelativeHumidityPercent,
    string? WindDirection,
    string? WindClass,
    string? Aqi,
    string? Pm25MicrogramsPerCubicMeter,
    string? VisibilityMeters,
    string? UpdatedAt);

internal sealed class BaiduWeatherApiException(
    int status,
    string message,
    bool isIpLocationRequest) : Exception(message)
{
    public int Status { get; } = status;

    public bool IsIpLocationRequest { get; } = isIpLocationRequest;
}
