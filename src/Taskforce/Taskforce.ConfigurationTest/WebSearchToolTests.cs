using System.Text.Json;
using DuckDuckGo.Net;
using Microsoft.Extensions.Logging;
using Moq;
using Taskforce.Core.Tools;

namespace Taskforce.ConfigurationTest
{
    [TestFixture]
    public class WebSearchToolTests
    {
        private WebSearchTool _webSearchTool;
        private Mock<ILogger> _loggerMock;
        private Mock<HttpMessageHandler> _httpMessageHandlerMock;

        [SetUp]
        public void Setup()
        {
            _loggerMock = new Mock<ILogger>();
            _httpMessageHandlerMock = new Mock<HttpMessageHandler>();
            _webSearchTool = new WebSearchTool(_loggerMock.Object);
        }

        [Test]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var tool = new WebSearchTool();

            // Assert
            Assert.That(tool, Is.Not.Null);
        }

        [Test]
        public void ExecuteSearch_WithValidJsonArgs_ThrowsInvalidOperationException_WhenEnvVarsNotSet()
        {
            // Arrange
            var searchArgs = new WebSearchArguments { Query = "test query" };
            var jsonArgs = JsonSerializer.Serialize(searchArgs);

            // Act & Assert
            Assert.ThrowsAsync<InvalidOperationException>(async () => 
                await _webSearchTool.Execute(jsonArgs));
        }

        [Test]
        public void ExecuteSearch_WithValidObjectArgs_ThrowsInvalidOperationException_WhenEnvVarsNotSet()
        {
            // Arrange
            var searchArgs = new WebSearchArguments { Query = "test query" };

            // Act & Assert
            Assert.ThrowsAsync<InvalidOperationException>(async () => 
                await _webSearchTool.Execute(searchArgs));
        }

        [Test]
        public void ExecuteSearch_WithInvalidJsonArgs_ThrowsArgumentException()
        {
            // Arrange
            var invalidJson = "{ invalid json }";

            // Act & Assert
            Assert.ThrowsAsync<ArgumentException>(async () => 
                await _webSearchTool.Execute(invalidJson));
        }

        [Test]
        public void ExecuteSearch_WithInvalidObjectArgs_ThrowsArgumentException()
        {
            // Arrange
            var invalidArgs = new object();

            // Act & Assert
            Assert.ThrowsAsync<ArgumentException>(async () => 
                await _webSearchTool.Execute(invalidArgs));
        }

        [Test]
        public void ExecuteSearch_WithEmptyQuery_ThrowsInvalidOperationException()
        {
            // Arrange
            var searchArgs = new WebSearchArguments { Query = string.Empty };
            var jsonArgs = JsonSerializer.Serialize(searchArgs);

            // Act & Assert
            Assert.ThrowsAsync<InvalidOperationException>(async () => 
                await _webSearchTool.Execute(jsonArgs));
        }

        [Test]
        public void ExecuteSearch_WithNullQuery_ThrowsInvalidOperationException()
        {
            // Arrange
            var searchArgs = new WebSearchArguments { Query = null };
            var jsonArgs = JsonSerializer.Serialize(searchArgs);

            // Act & Assert
            Assert.ThrowsAsync<InvalidOperationException>(async () => 
                await _webSearchTool.Execute(jsonArgs));
        }
    }
} 