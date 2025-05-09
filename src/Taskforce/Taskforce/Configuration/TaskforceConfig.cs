﻿using Taskforce.Domain.Entities;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace Taskforce.Configuration
{
    public class TaskforceConfig
    {
        public List<AgentConfig> AgentConfigs { get; set; }

        public static TaskforceConfig Create(string configFile)
        {
            if (!File.Exists(configFile))
            { 
                throw new FileNotFoundException(configFile);
            }

            var configContent = File.ReadAllText(configFile);
            var deserializer = new DeserializerBuilder().Build();
            var config = deserializer.Deserialize<TaskforceConfig>(configContent);

            return config;
        }
    }
}
