namespace Taskforce.Domain.Entities
{
    public class AgentConfig
    {
        public string Name { get; set; } = string.Empty;

        public string Role { get; set; } = string.Empty;

        public string Mission { get; set; } = string.Empty;

        public bool WithVision { get; set; } = false;
    }
}