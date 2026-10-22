using System.Text.Json;
using System.Text.RegularExpressions;
using VoiceAssistantDemo.Tools.BaiduApiClient;
using VoiceBox.Assistant.Api.Tools.Contracts;
using VoiceBox.Assistant.Api.Tools.Models;

namespace VoiceAssistantDemo.Tools;

/// <summary>
/// Gets current weather from Baidu Maps. A supplied administrative name is resolved through the
/// embedded district catalog; an omitted location falls back to public-IP geolocation.
/// </summary>
internal sealed partial class CurrentWeatherTool : IAssistantTool
{
    private static readonly HttpClient SharedHttpClient = new()
    {
        Timeout = TimeSpan.FromSeconds(15),
    };
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    private readonly BaiduWeatherClient _client;
    private readonly WeatherDistrictCatalog _districts;

    public CurrentWeatherTool()
        : this(new BaiduWeatherClient(SharedHttpClient), WeatherDistrictCatalog.Shared)
    {
    }

    internal CurrentWeatherTool(
        BaiduWeatherClient client,
        WeatherDistrictCatalog districts)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _districts = districts ?? throw new ArgumentNullException(nameof(districts));
    }

    public AssistantToolDescriptor Descriptor { get; } = new(
        name: "get_current_weather",
        displayName: "获取当前天气",
        description:
            "获取指定位置的当前天气；location 未提供时，使用请求设备的公网 IP 进行大致定位。" +
            "支持城市、区县、行政区划 ID，以及“经度,纬度”。",
        inputSchemaJson:
            """
            {
              "type": "object",
              "properties": {
                "location": {
                  "type": "string",
                  "description": "可选。城市或区县名称、至少 6 位行政区划 ID，或“经度,纬度”；省略时使用 IP 定位。"
                },
                "province": {
                  "type": "string",
                  "description": "可选。名称重名时用于消歧的省份，例如“吉林省”。"
                },
                "city": {
                  "type": "string",
                  "description": "可选。名称重名时用于消歧的城市，例如“长春市”。"
                },
                "coordinateType": {
                  "type": "string",
                  "enum": ["wgs84", "gcj02", "bd09ll", "bd09mc"],
                  "description": "location 为经纬度时的坐标类型，默认 wgs84。"
                }
              },
              "additionalProperties": false
            }
            """,
        outputSchemaJson:
            """
            {
              "type": "object",
              "properties": {
                "positionSource": { "type": "string", "enum": ["specified", "ip"] },
                "queryPosition": { "type": "string" },
                "location": { "type": "string" },
                "districtId": { "type": ["string", "null"] },
                "weather": { "type": "string" },
                "temperatureCelsius": { "type": ["string", "null"] },
                "feelsLikeCelsius": { "type": ["string", "null"] },
                "relativeHumidityPercent": { "type": ["string", "null"] },
                "windDirection": { "type": ["string", "null"] },
                "windClass": { "type": ["string", "null"] },
                "aqi": { "type": ["string", "null"] },
                "pm25MicrogramsPerCubicMeter": { "type": ["string", "null"] },
                "visibilityMeters": { "type": ["string", "null"] },
                "updatedAt": { "type": ["string", "null"] }
              },
              "required": [
                "positionSource",
                "queryPosition",
                "location",
                "districtId",
                "weather",
                "temperatureCelsius",
                "feelsLikeCelsius",
                "relativeHumidityPercent",
                "windDirection",
                "windClass",
                "aqi",
                "pm25MicrogramsPerCubicMeter",
                "visibilityMeters",
                "updatedAt"
              ],
              "additionalProperties": false
            }
            """,
        risk: AssistantToolRisk.ReadOnly,
        executionPolicy: AssistantToolExecutionPolicy.BackgroundOnly,
        executionPrompt: "正在查询，请稍后",
        timeout: TimeSpan.FromSeconds(20));

    public async ValueTask<AssistantToolExecutionResult> ExecuteAsync(
        AssistantToolInvocation invocation,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(invocation);
        try
        {
            WeatherToolArguments arguments = ParseArguments(invocation.Arguments);
            WeatherPosition position = await ResolvePositionAsync(arguments, cancellationToken)
                .ConfigureAwait(false);
            CurrentWeatherReport report = await _client.QueryCurrentWeatherAsync(
                position,
                cancellationToken).ConfigureAwait(false);
            return AssistantToolExecutionResult.Success(
                CreateSpokenSummary(report),
                JsonSerializer.Serialize(report, JsonOptions));
        }
        catch (WeatherLocationException exception)
        {
            return AssistantToolExecutionResult.Error("weather_location_invalid", exception.Message);
        }
        catch (BaiduWeatherApiException exception)
        {
            string message = exception.IsIpLocationRequest
                ? "无法通过公网 IP 确定当前位置，请让用户明确提供城市或区县。"
                : $"天气服务暂时不可用（状态码 {exception.Status}）。";
            return AssistantToolExecutionResult.Error("weather_api_error", message);
        }
        catch (HttpRequestException)
        {
            return AssistantToolExecutionResult.Error(
                "weather_network_error",
                "天气服务网络请求失败，请稍后重试。");
        }
        catch (JsonException)
        {
            return AssistantToolExecutionResult.Error(
                "weather_response_invalid",
                "天气服务返回了无法识别的数据。");
        }
    }

    private async Task<WeatherPosition> ResolvePositionAsync(
        WeatherToolArguments arguments,
        CancellationToken cancellationToken)
    {
        string? location = arguments.Location;
        if (string.IsNullOrWhiteSpace(location))
            location = arguments.City ?? arguments.Province;
        if (string.IsNullOrWhiteSpace(location))
            return await _client.LocateCurrentIpAsync(cancellationToken).ConfigureAwait(false);

        string target = location.Trim();
        if (DistrictIdRegex().IsMatch(target))
            return WeatherPosition.ForDistrict(target, $"行政区划 ID {target}");
        if (target.Contains(','))
        {
            string coordinateType = CoordinateParser.ValidateType(arguments.CoordinateType);
            string coordinates = CoordinateParser.Normalize(target, coordinateType);
            return WeatherPosition.ForCoordinate(
                coordinates,
                coordinateType,
                $"经纬度 {coordinates}（{coordinateType}）");
        }

        DistrictRecord district = _districts.Resolve(
            target,
            arguments.Province,
            arguments.City);
        return WeatherPosition.ForDistrict(
            district.DistrictId,
            $"{district.Province} {district.City} {district.District}");
    }

    private static WeatherToolArguments ParseArguments(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
            throw new WeatherLocationException("天气工具参数必须是 JSON 对象。");
        return new WeatherToolArguments(
            ReadOptionalString(root, "location"),
            ReadOptionalString(root, "province"),
            ReadOptionalString(root, "city"),
            ReadOptionalString(root, "coordinateType"));
    }

    private static string? ReadOptionalString(JsonElement root, string propertyName)
    {
        if (!root.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind == JsonValueKind.Null)
        {
            return null;
        }
        if (value.ValueKind != JsonValueKind.String)
            throw new WeatherLocationException($"参数 {propertyName} 必须是字符串。");
        return string.IsNullOrWhiteSpace(value.GetString()) ? null : value.GetString()!.Trim();
    }

    private static string CreateSpokenSummary(CurrentWeatherReport report)
    {
        var details = new List<string> { $"{report.Location}当前{report.Weather}" };
        if (report.TemperatureCelsius is not null)
            details.Add($"温度{report.TemperatureCelsius}摄氏度");
        if (report.FeelsLikeCelsius is not null)
            details.Add($"体感{report.FeelsLikeCelsius}摄氏度");
        if (report.RelativeHumidityPercent is not null)
            details.Add($"湿度百分之{report.RelativeHumidityPercent}");
        string wind = string.Join(
            ' ',
            new[] { report.WindDirection, report.WindClass }
                .Where(static value => !string.IsNullOrWhiteSpace(value)));
        if (wind.Length > 0)
            details.Add(wind);
        if (report.Aqi is not null)
            details.Add($"空气质量指数{report.Aqi}");
        if (report.PositionSource == "ip")
            details.Add("位置由公网 IP 大致确定");
        return string.Join("，", details) + "。";
    }

    [GeneratedRegex("^\\d{6,}$", RegexOptions.CultureInvariant)]
    private static partial Regex DistrictIdRegex();

    public void Dispose() { }

    private sealed record WeatherToolArguments(
        string? Location,
        string? Province,
        string? City,
        string? CoordinateType);
}
