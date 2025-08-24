from crewai import BaseLLM
import replicate

class ReplicateChatLLM(BaseLLM):
    """Wrapper to use Replicate-hosted chat models with CrewAI (plain text I/O)."""
    def __init__(self, model: str | None = None, temperature: float | None = 0.0, extra_inputs: dict | None = None):
        super().__init__(model=model or "openai/gpt-oss-20b", temperature=temperature)
        self.extra_inputs = extra_inputs or {}

    def _messages_to_prompt(self, messages):
        if isinstance(messages, str):
            return messages
        parts = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            parts.append(f"{role}:\n{content}")
        return "\n\n".join(parts)

    def call(self, messages, **kwargs) -> str:
        prompt = self._messages_to_prompt(messages)
        preamble = (
            "You are a careful bias-detection analyst.\n"
            "Reasoning: high\n"
            "Return ONLY valid JSON for the requested schema. No prose."
        )
        payload = {"prompt": f"{preamble}\n\n{prompt}"}
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)
        payload.update(self.extra_inputs or {})
        resp = replicate.run(self.model, input=payload)
        if hasattr(resp, "__iter__") and not isinstance(resp, (str, bytes)):
            return "".join(chunk for chunk in resp)
        return str(resp)

    def supports_stop_words(self) -> bool:
        return False

    def __call__(self, messages, **kwargs):
        return self.call(messages, **kwargs)
