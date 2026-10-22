namespace VoiceBox.Assistant.Runtime.Lifecycle.Models;

internal readonly record struct AssistantTaskRegistration(
        string Name,
        long? OwnerTurnId,
        AssistantTaskFaultPolicy FaultPolicy);
