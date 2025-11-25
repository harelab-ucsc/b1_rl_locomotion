import carb
import numpy as np
import omni
import omni.appwindow  # Contains handle to keyboard
from isaacsim.examples.interactive.base_sample import BaseSample
from scripts.b1_policy import B1StandPolicy

# from b1_rl_locomotion.b1_rl_locomotion.tasks.manager_based.b1_rl_locomotion.b1_rl_locomotion_env_cfg import SpotFlatTerrainPolicy

class B1Stand(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 500.0
        self._world_settings["rendering_dt"] = 10.0 / 500.0
        self._base_command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]

        # bindings for keyboard to command
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [0.0, 0.0, 0.54, 0, 0, 0, 0],    # Height Command
            "UP": [0.0, 0.0, 0.54, 0, 0, 0, 0],          # Height Command
        }

    def setup_scene(self) -> None:
        self._world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
        )
        self.spot = B1StandPolicy(
            prim_path="/Robot",
            name="b1_description",
            position=np.array([0, 0, 0.12]),
        )
        timeline = omni.timeline.get_timeline_interface()
        self._event_timer_callback = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), self._timeline_timer_callback_fn
        )

    async def setup_post_load(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._physics_ready = False
        self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        await self.get_world().play_async()

    async def setup_post_reset(self) -> None:
        self._physics_ready = False
        await self._world.play_async()

    def on_physics_step(self, step_size) -> None:
        if self._physics_ready:
            self.spot.stand(step_size, self._base_command)
        else:
            self._physics_ready = True
            self.spot.initialize()
            self.spot.post_reset()
            self.spot.robot.set_joints_default_state(self.spot.default_pos)

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """Subscriber callback to when kit is updated."""

        # when a key is pressedor released  the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # on pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])

    def _timeline_timer_callback_fn(self, event) -> None:
        if self.spot:
            self._physics_ready = False

    def world_cleanup(self):
        self._event_timer_callback = None
        if self._world.physics_callback_exists("physics_step"):
            self._world.remove_physics_callback("physics_step")
