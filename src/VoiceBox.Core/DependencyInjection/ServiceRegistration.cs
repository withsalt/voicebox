using System.Diagnostics.CodeAnalysis;

namespace VoiceBox.Core;

public sealed class ServiceRegistration<
    TService,
    [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)] TImplementation>
    where TService : class
    where TImplementation : class, TService
{
    private readonly AudioServiceManager _manager;
    private readonly string? _name;

    internal ServiceRegistration(AudioServiceManager manager, string? name)
    {
        _manager = manager;
        _name = name;
    }

    public AudioServiceManager Configure<TOptions>(Action<TOptions>? configure = null)
        where TOptions : class
    {
        return _manager.AddService<TService, TImplementation, TOptions>(configure, _name);
    }
}
