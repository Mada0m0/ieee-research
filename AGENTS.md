# AI Agents Responsibilities

This platform uses multiple AI agents to work together to complete complex control algorithm research tasks.

## Hermes
- **Main Roles**: System Coordinator and Mission Planner
- **Core Responsibilities**:
  - Analyze the user's macro needs and break them into executable subtasks.
  - Manage task queues and distribute subtasks to appropriate agents for processing.
  - Monitor the task execution of each agent and coordinate the flow of resources and information between agents.

## Claude
- **Main Role**: Theoretical Researcher and Algorithm Designer
- **Core Responsibilities**:
  - Based on IEEE related literature and control theory, the theoretical framework and derivation of the algorithm are provided.
  - Assist in writing in-depth technical documentation and research reports.
  - Perform theoretical analysis and evaluation of complex control systems.

## Jules
- **Main Roles**: Core Software Engineers and Executors
- **Core Responsibilities**:
  - Convert the algorithms designed by Claude into high-quality, runnable code.
  - Responsible for code refactoring, testing, debugging and performance optimization.
  - Manage the code warehouse and execute CI/CD related scripts and automated maintenance tasks.
