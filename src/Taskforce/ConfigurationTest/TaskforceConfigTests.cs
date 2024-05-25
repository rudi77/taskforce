using Configuration;

namespace ConfigurationTest
{
    public class TaskforceConfigTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Create_Should_Return_TaskforceConfig_Object()
        {
            var config = TaskforceConfig.Create("./sample/taskforce.yaml");
            Assert.IsNotNull(config);
        }
    }
}