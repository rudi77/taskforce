using Newtonsoft.Json;

namespace Planning
{
    internal class Questions
    {
        [JsonProperty("sub-question")]
        public List<string> SubQuestions { get; set; }
    }
}
