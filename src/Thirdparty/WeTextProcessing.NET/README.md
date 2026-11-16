# WeTextProcessing.NET

A .NET wrapper for [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing), supporting Chinese and English text normalization (TN) and inverse text normalization (ITN).
The package includes native libraries and FST models built from WeTextProcessing 1.3.0 with default options.

## Installation

```bash
dotnet add package WeTextProcessing.NET
```

Supports .NET 8, 9, 10, and 11. Native libraries are included for Windows x64, Linux x64, and Linux arm64. Linux requires glibc 2.38 or later.

## Quick Start

```csharp
using System;
using WeTextProcessing.NET;

// Chinese TN: convert written text to spoken form. Chinese is the default language.
using var chinese = new WetextProcessor();
Console.WriteLine(chinese.Normalize("2023年5月18日"));
// 二零二三年五月十八日

// Chinese ITN: convert spoken form to written text.
using var inverse = new WetextProcessor(WetextLanguage.Chinese, WetextMode.InverseTextNormalization);
Console.WriteLine(inverse.Normalize("一百二十三"));
// 123

// English TN. For English ITN, use WetextMode.InverseTextNormalization.
using var english = new WetextProcessor(WetextLanguage.English);
Console.WriteLine(english.Normalize("1000"));
// one thousand
```

`WetextProcessor` loads models when constructed. Reuse instances where possible, and release resources with `using` or `Dispose()` when finished.
Use `Tag()` to produce tagged text or `Verbalize()` to convert tagged text to its final form.

## Deployment

The NuGet build targets copy models and native libraries to the output directory. Keep the `runtimes` directory when deploying. Default models are loaded from `runtimes/models/WeTextProcessing` under the application directory.
To load custom models, use `new WetextProcessor(taggerPath, verbalizerPath)`. Keep the original `zh_tn_`, `zh_itn_`, `en_tn_`, or `en_itn_` filename prefix.

## Project and License

- [.NET wrapper source](https://github.com/withsalt/voicebox/tree/main/src/Thirdparty/WeTextProcessing.NET)
- [Upstream WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)
- License: Apache-2.0.
