namespace VoiceBox.Assistant.Modules.Speaker.Models;

public enum SpeakerEnrollmentStage
{
    PreparingFrontend,
    CleanEnrollment,
    DoubleTalkEnrollment,
    CleanConfirmation,
    DoubleTalkConfirmation,
    BuildingProfile,
    SavingProfile,
    Completed,
    Failed,
}
