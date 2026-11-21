class History:
    """Conversation memory, trimmed to the last N user/assistant turns."""

    def __init__(self, system_prompt: str, max_turns: int):
        self.system = {"role": "system", "content": system_prompt}
        self.max_messages = max_turns * 2
        self.turns: list[dict] = []

    def add_user(self, text: str):
        self.turns.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str):
        if text:
            self.turns.append({"role": "assistant", "content": text})
            self._trim()

    def _trim(self):
        if len(self.turns) > self.max_messages:
            self.turns = self.turns[-self.max_messages :]

    def messages(self) -> list[dict]:
        return [self.system, *self.turns]

    def clear(self):
        self.turns.clear()
