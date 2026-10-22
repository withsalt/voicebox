using System.Text.Json.Nodes;
using Microsoft.Extensions.Options;
using VoiceBox.Wakeup.FsmnKws.Models;
using Xunit;

namespace VoiceBox.Wakeup.FsmnKws.Tests;

public sealed class FsmnKwsMetadataSpecs
{
    [Fact]
    public void LoadsEmbeddedCmvn()
    {
        using var file = new ModelMetadataFile(CreateMetadata());

        FsmnKwsMetadata metadata = FsmnKwsMetadata.Load(file.Path);

        Assert.Equal("小爱同学", metadata.Keyword);
        Assert.Equal(new[] { 1462, 1947, 1218, 1428 }, metadata.KeywordTokenIds);
        Assert.Equal(400, metadata.CmvnAddShift.Count);
        Assert.Equal(400, metadata.CmvnRescale.Count);
        Assert.All(metadata.CmvnAddShift, value => Assert.Equal(-2.5f, value));
        Assert.All(metadata.CmvnRescale, value => Assert.Equal(0.375f, value));
    }

    [Theory]
    [InlineData("add_shift", 0)]
    [InlineData("add_shift", 399)]
    [InlineData("add_shift", 401)]
    [InlineData("rescale", 0)]
    [InlineData("rescale", 399)]
    [InlineData("rescale", 401)]
    public void RejectsWrongCmvnDimensions(string field, int count)
    {
        JsonObject metadata = CreateMetadata();
        metadata["cmvn"]![field] = CreateValues(count, 1.0f);
        using var file = new ModelMetadataFile(metadata);

        var error = Assert.Throws<InvalidDataException>(() => FsmnKwsMetadata.Load(file.Path));

        Assert.Contains($"cmvn.{field}", error.Message);
        Assert.Contains("400", error.Message);
    }

    [Theory]
    [InlineData("add_shift", "null")]
    [InlineData("add_shift", "true")]
    [InlineData("add_shift", "\"bad\"")]
    [InlineData("add_shift", "1e100")]
    [InlineData("rescale", "null")]
    [InlineData("rescale", "1e100")]
    [InlineData("rescale", "0")]
    [InlineData("rescale", "-1")]
    public void RejectsInvalidCmvnValues(string field, string value)
    {
        JsonObject metadata = CreateMetadata();
        metadata["cmvn"]![field]![10] = JsonNode.Parse(value);
        using var file = new ModelMetadataFile(metadata);

        var error = Assert.Throws<InvalidDataException>(() => FsmnKwsMetadata.Load(file.Path));

        Assert.Contains($"cmvn.{field}[10]", error.Message);
    }

    [Theory]
    [InlineData("null")]
    [InlineData("[]")]
    [InlineData("1")]
    public void RejectsInvalidCmvnObjects(string value)
    {
        JsonObject metadata = CreateMetadata();
        metadata["cmvn"] = JsonNode.Parse(value);
        using var file = new ModelMetadataFile(metadata);

        Assert.Throws<InvalidDataException>(() => FsmnKwsMetadata.Load(file.Path));
    }

    [Theory]
    [InlineData("add_shift")]
    [InlineData("rescale")]
    public void RejectsMissingCmvnArrays(string field)
    {
        JsonObject metadata = CreateMetadata();
        metadata["cmvn"]!.AsObject().Remove(field);
        using var file = new ModelMetadataFile(metadata);

        var error = Assert.Throws<InvalidDataException>(() => FsmnKwsMetadata.Load(file.Path));

        Assert.Contains($"cmvn.{field}", error.Message);
    }

    [Fact]
    public void MissingModelIsReportedAfterLoadingEmbeddedCmvn()
    {
        using var file = new ModelMetadataFile(CreateMetadata());
        string missingModel = System.IO.Path.ChangeExtension(file.Path, ".onnx");
        var options = new FsmnKwsOptions { ModelPath = missingModel };

        var error = Assert.Throws<FileNotFoundException>(() => new FsmnKwsService(Options.Create(options)));

        Assert.Equal(missingModel, error.FileName);
    }

    [Fact]
    public void MissingCmvnIsRejectedEvenWhenAmMvnExists()
    {
        JsonObject metadata = CreateMetadata();
        metadata.Remove("cmvn");
        using var file = new ModelMetadataFile(metadata);
        string values = string.Join(' ', Enumerable.Repeat("1", 400));
        File.WriteAllText(System.IO.Path.Combine(file.DirectoryPath, "am.mvn"),
            $"<AddShift> 400 400\n<LearnRateCoef> 0 [ {values} ]\n"
            + $"<Rescale> 400 400\n<LearnRateCoef> 0 [ {values} ]\n");
        var options = new FsmnKwsOptions { ModelPath = System.IO.Path.ChangeExtension(file.Path, ".onnx") };

        var error = Assert.Throws<InvalidDataException>(() => new FsmnKwsService(Options.Create(options)));

        Assert.Contains("kws_meta.json field 'cmvn'", error.Message);
    }

    private static JsonObject CreateMetadata() => new()
    {
        ["keyword"] = "小爱同学",
        ["keyword_token_ids"] = new JsonArray(1462, 1947, 1218, 1428),
        ["cmvn"] = new JsonObject
        {
            ["add_shift"] = CreateValues(400, -2.5f),
            ["rescale"] = CreateValues(400, 0.375f),
        },
    };

    private static JsonArray CreateValues(int count, float value) =>
        new(Enumerable.Range(0, count).Select(_ => (JsonNode?)JsonValue.Create(value)).ToArray());

    private sealed class ModelMetadataFile : IDisposable
    {
        public string DirectoryPath { get; } = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), $"fsmnkws-meta-{Guid.NewGuid():N}");
        public string Path => System.IO.Path.Combine(DirectoryPath, "kws_meta.json");

        public ModelMetadataFile(JsonObject metadata)
        {
            Directory.CreateDirectory(DirectoryPath);
            File.WriteAllText(Path, metadata.ToJsonString());
        }

        public void Dispose()
        {
            foreach (string file in Directory.GetFiles(DirectoryPath))
                File.Delete(file);
            Directory.Delete(DirectoryPath);
        }
    }
}
