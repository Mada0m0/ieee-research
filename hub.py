import logging
from typing import Dict, Any, List

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IEEE_Control_Hub")

class Task:
    """Represents a research task in the hub."""
    def __init__(self, task_id: str, description: str, assignee: str = None):
        self.task_id = task_id
        self.description = description
        self.assignee = assignee
        self.status = "Pending"

    def __repr__(self):
        return f"<Task {self.task_id} | Status: {self.status} | Assignee: {self.assignee}>"

class CentralHub:
    """Central scheduling and dispatch module for the IEEE Control Algorithm Research Hub."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.agents = ["Hermes", "Claude", "Jules"]
        logger.info("Central Hub Initialized. Available Agents: %s", ", ".join(self.agents))

    def create_task(self, task_id: str, description: str) -> Task:
        """Creates a new task and adds it to the hub."""
        if task_id in self.tasks:
            logger.warning("Task ID %s already exists.", task_id)
            return self.tasks[task_id]

        new_task = Task(task_id, description)
        self.tasks[task_id] = new_task
        logger.info("Created Task: %s", task_id)
        return new_task

    def dispatch_task(self, task_id: str, agent_name: str) -> bool:
        """Assigns a task to a specific AI agent."""
        if task_id not in self.tasks:
            logger.error("Task ID %s not found.", task_id)
            return False

        if agent_name not in self.agents:
            logger.error("Agent %s is not registered. Available agents: %s", agent_name, self.agents)
            return False

        task = self.tasks[task_id]
        task.assignee = agent_name
        task.status = "In Progress"
        logger.info("Dispatched Task %s to Agent %s", task_id, agent_name)
        return True

    def complete_task(self, task_id: str) -> bool:
        """Marks a task as completed."""
        if task_id not in self.tasks:
            logger.error("Task ID %s not found.", task_id)
            return False

        task = self.tasks[task_id]
        task.status = "Completed"
        logger.info("Task %s completed by %s", task_id, task.assignee)
        return True

    def list_tasks(self) -> List[Task]:
        """Returns a list of all tasks."""
        return list(self.tasks.values())

def main():
    """Main execution block for demonstration."""
    hub = CentralHub()

    # Example workflow
    hub.create_task("T-001", "Analyze IEEE standard for linear controllers")
    hub.create_task("T-002", "Implement PID controller algorithm")

    hub.dispatch_task("T-001", "Claude")
    hub.dispatch_task("T-002", "Jules")

    hub.complete_task("T-001")

    print("\nCurrent Hub Status:")
    for task in hub.list_tasks():
        print(f" - {task}")

if __name__ == "__main__":
    main()
