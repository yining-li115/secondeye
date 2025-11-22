from pathlib import Path
from typing import Dict, Tuple
import math

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.audio.tts import text_to_speech_gentle
from app.config import settings


class Position3D:
    """Represents a 3D position with x, y, z coordinates."""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def distance_to(self, other: 'Position3D') -> float:
        """Calculate Euclidean distance to another position."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def horizontal_distance_to(self, other: 'Position3D') -> float:
        """Calculate horizontal (x-z plane) distance to another position."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.z - other.z) ** 2
        )

    def __repr__(self):
        return f"Position3D(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


class Orientation:
    """Represents orientation with yaw (horizontal rotation) and pitch (vertical tilt)."""
    def __init__(self, yaw: float, pitch: float = 0.0):
        self.yaw = yaw  # Rotation around Y-axis (in degrees, 0=north, 90=east)
        self.pitch = pitch  # Vertical tilt (in degrees)

    def __repr__(self):
        return f"Orientation(yaw={self.yaw:.1f} deg, pitch={self.pitch:.1f} deg)"


def calculate_navigation_direction(
    current_pos: Position3D,
    current_orientation: Orientation,
    target_pos: Position3D
) -> Dict:
    """
    Calculate navigation direction from current position to target.

    Args:
        current_pos: Current 3D position
        current_orientation: Current orientation (yaw and pitch)
        target_pos: Target 3D position

    Returns:
        Dict with navigation metrics:
        - distance: Total 3D distance to target
        - horizontal_distance: Horizontal distance in x-z plane
        - vertical_distance: Vertical distance (y-axis)
        - relative_angle: Angle to target relative to current orientation (-180 to 180)
        - direction: Simple direction (forward, left, right, back)
        - height_diff: Height difference (positive = target is higher)
    """
    # Calculate distances
    total_distance = current_pos.distance_to(target_pos)
    horizontal_distance = current_pos.horizontal_distance_to(target_pos)
    vertical_distance = target_pos.y - current_pos.y

    # Calculate angle to target in x-z plane
    dx = target_pos.x - current_pos.x
    dz = target_pos.z - current_pos.z
    angle_to_target = math.degrees(math.atan2(dx, dz))  # atan2(x, z) for yaw

    # Calculate relative angle to current orientation
    relative_angle = angle_to_target - current_orientation.yaw
    # Normalize to -180 to 180
    while relative_angle > 180:
        relative_angle -= 360
    while relative_angle < -180:
        relative_angle += 360

    # Determine simple direction
    if abs(relative_angle) < 45:
        direction = "forward"
    elif abs(relative_angle) > 135:
        direction = "behind"
    elif relative_angle > 0:
        direction = "right"
    else:
        direction = "left"

    return {
        "distance": total_distance,
        "horizontal_distance": horizontal_distance,
        "vertical_distance": vertical_distance,
        "relative_angle": relative_angle,
        "direction": direction,
        "height_diff": vertical_distance
    }


def generate_navigation_instructions(
    target_object: str,
    nav_metrics: Dict,
    current_pos: Position3D,
    target_pos: Position3D
) -> str:
    """
    Use LLM to generate gentle, patient navigation instructions.

    Args:
        target_object: Name of the target object (e.g., "apple")
        nav_metrics: Navigation metrics from calculate_navigation_direction()
        current_pos: Current position
        target_pos: Target position

    Returns:
        Gentle, conversational navigation instructions
    """
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )

    system_prompt = (
        "You are a navigation assistant for visually impaired users. "
        "Generate VERY SHORT and CLEAR navigation instructions. "
        "Maximum 2-3 sentences. Be direct and concise.\n\n"
        "Format: [Direction], [Distance], [Height if relevant].\n"
        "Example: 'Turn left 15 degrees. Walk forward 3 steps. The cup is at waist height.'"
    )

    distance = nav_metrics["horizontal_distance"]
    direction = nav_metrics["direction"]
    relative_angle = nav_metrics["relative_angle"]
    height_diff = nav_metrics["height_diff"]

    # Convert distance to steps (rough approximation: 1 meter ~= 1.5 steps)
    steps = int(distance * 1.5)

    user_prompt = (
        f"Generate SHORT navigation instructions to reach a {target_object}.\n\n"
        f"Details:\n"
        f"- Distance: {distance:.1f} meters ({steps} steps)\n"
        f"- Turn: {relative_angle:.0f} degrees ({direction})\n"
        f"- Height: {height_diff:.1f}m ({'higher' if height_diff > 0 else 'lower' if height_diff < 0 else 'same level'})\n\n"
        f"Keep it to 2-3 short sentences maximum. Be clear and direct."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])

    chain = prompt | llm
    response = chain.invoke({})

    return response.content


def navigate_to_target(
    target_object: str,
    current_pos: Position3D,
    current_orientation: Orientation,
    target_pos: Position3D,
    output_audio_path: Path
) -> Dict:
    """
    Agent 3 main workflow:
    1. Calculate navigation metrics based on 3D positions
    2. Generate gentle navigation instructions using LLM
    3. Convert to speech with TTS

    Args:
        target_object: Name of target object (e.g., "apple")
        current_pos: User's current 3D position
        current_orientation: User's current orientation
        target_pos: Target object's 3D position
        output_audio_path: Where to save TTS audio output

    Returns:
        Dict with navigation info and audio output path
    """
    # 1. Calculate navigation metrics
    nav_metrics = calculate_navigation_direction(
        current_pos,
        current_orientation,
        target_pos
    )

    # 2. Generate gentle navigation instructions using LLM
    instructions = generate_navigation_instructions(
        target_object,
        nav_metrics,
        current_pos,
        target_pos
    )

    # 3. Convert to gentle speech
    text_to_speech_gentle(instructions, output_audio_path)

    return {
        "target_object": target_object,
        "current_position": str(current_pos),
        "target_position": str(target_pos),
        "navigation_metrics": nav_metrics,
        "instructions": instructions,
        "audio_output": str(output_audio_path)
    }


def create_step_by_step_guidance(
    target_object: str,
    steps_remaining: int,
    direction: str
) -> str:
    """
    Generate step-by-step guidance with counting for continuous navigation.

    Args:
        target_object: Target object name
        steps_remaining: Number of steps remaining
        direction: Current direction to maintain

    Returns:
        Simple guidance text for current step
    """
    if steps_remaining > 5:
        return f"Keep going {direction}. {steps_remaining} steps to the {target_object}. I'm here with you."
    elif steps_remaining > 0:
        return f"Almost there! Just {steps_remaining} more steps {direction}."
    else:
        return f"You've arrived! The {target_object} should be right in front of you now."
