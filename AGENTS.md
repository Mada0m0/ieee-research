# AI Agents Responsibilities

This platform uses multiple AI agents to work together to complete complex control algorithm research tasks.

## Hermes
- **Main Roles**: System Coordinator and Mission Planner
- **Core Responsibilities**:
  - Analyze users' macro needs and break them into executable sub-tasks.
  - Manage task queues and distribute subtasks to appropriate agents for processing.
  - Monitor the task execution of each agent and coordinate the flow of resources and information between agents.

## Claude
- **Main Role**: Theoretical Researcher and Algorithm Designer
- **Core Responsibilities**:
  - Based on IEEE related literature and control theory, provide the theoretical framework and derivation of the algorithm.
  - Assist in writing in-depth technical documentation and research reports.
  - Perform theoretical analysis and evaluation of complex control systems.

## Jules
- **Main Roles**: Core Software Engineers and Executors
- **Core Responsibilities**:
  - Convert the algorithms designed by Claude into high-quality, runnable code.
  - Responsible for code refactoring, testing, debugging and performance optimization.
  - Manage code warehouse, execute CI/CD related scripts and automated maintenance tasks.
