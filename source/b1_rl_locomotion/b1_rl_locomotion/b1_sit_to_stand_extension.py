import os

import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from .b1_sit_to_stand import B1Stand



class B1StandExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.name = "B1 stand"
        self.category = "Policy"

        overview = "B1 running sit->stand policy"
        overview += "\n\tKeybord Input:"
        overview += "\n\t\tup arrow / numpad 8: stand up"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Unitree: B1",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_lab_tutorials/tutorial_policy_deployment.html",
            "overview": overview,
            "sample": B1Stand(),
        }

        ui_handle = BaseSampleUITemplate(**ui_kwargs)

        # register the example with examples browser
        get_browser_instance().register_example(
            name=self.name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

        return

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.name, category=self.category)

        return
