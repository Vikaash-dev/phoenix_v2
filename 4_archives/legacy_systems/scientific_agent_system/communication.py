from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import time

@dataclass
class Message:
    sender: str
    recipient: str
    content: str
    message_type: str  # 'instruction', 'report', 'query', 'response'
    timestamp: float = 0.0

    def __post_init__(self):
        self.timestamp = time.time()

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(json_str: str) -> 'Message':
        data = json.loads(json_str)
        return Message(**data)

class CommunicationChannel:
    def __init__(self):
        self.history: List[Message] = []

    def send(self, message: Message):
        print(f"[{message.sender} -> {message.recipient}] ({message.message_type}): {message.content[:100]}...")
        self.history.append(message)

    def get_history(self) -> List[Message]:
        return self.history

    def get_messages_for(self, recipient: str) -> List[Message]:
        return [m for m in self.history if m.recipient == recipient]
