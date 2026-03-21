import os
import webbrowser

import cv2
import imgui
import time
import gc

from assets import RED
from utils.gui_utils import imgui_window, gl_utils
from utils.gui_utils import imgui_utils
from utils.version import get_version
from widgets.help_icon_widget import DOCS_BASE_URL
from enum import IntEnum


class States(IntEnum):
    ERROR = -2
    CLOSE = -1
    WELCOME = 0
    LIVE = 1
    SPLASH = 3
    PREPROCESSING = 4
    TRAINING = 5
    TOOLS = 6


class ModuleHost:
    def __init__(self, app):
        self.app = app


class Autolume(imgui_window.ImguiWindow):
    def __init__(self):
        super().__init__(title=f'Autolume-Live v{get_version()}', window_width=3840, window_height=2160)

        self.state = States.WELCOME
        self.running = True
        self.viz = None
        self.render_loop = None
        self.pkls = []
        self.splash_delay = 0
        self.data_preprocessing = None

        self._training_module = None
        self._projection_module = None
        self._features_module = None
        self._super_res_module = None
        self._mixing_module = None

        self.splash = cv2.imread("assets/splashscreen.jpg", cv2.IMREAD_UNCHANGED)
        self.splash = cv2.cvtColor(self.splash, cv2.COLOR_BGRA2RGBA)
        self.splash_texture = gl_utils.Texture(image=self.splash, width=self.splash.shape[1],
                                               height=self.splash.shape[0], channels=self.splash.shape[2])

        self.logo = cv2.imread("assets/Autolume-logo.png", cv2.IMREAD_UNCHANGED)
        self.logo_texture = gl_utils.Texture(image=self.logo, width=self.logo.shape[1],
                                             height=self.logo.shape[0], channels=self.logo.shape[2])
        self.metacreation = cv2.imread("assets/metalogo.png", cv2.IMREAD_UNCHANGED)
        self.metacreation_texture = gl_utils.Texture(image=self.metacreation, width=self.metacreation.shape[1],
                                                     height=self.metacreation.shape[0],
                                                     channels=self.metacreation.shape[2])
        self.navbar_height = 50

        self.label_w = 0
        self.button_w = 0
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame()

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame()

    def close(self):
        if self.data_preprocessing is not None:
            self.data_preprocessing.cleanup()
            self.data_preprocessing = None

        super().close()

    def _cleanup_current_state(self):
        if self.viz is not None:
            self.viz.close()
            self.viz = None
        if self.render_loop is not None:
            self.render_loop.close()
            self.render_loop = None
        if self.data_preprocessing is not None:
            self.data_preprocessing.cleanup()
            self.data_preprocessing = None
        gc.collect()
        time.sleep(0.05)

    def navigate_to(self, target_state):
        if target_state == self.state:
            return

        if self._is_training_active() and self.state == States.TRAINING:
            return

        self._cleanup_current_state()

        if target_state == States.LIVE:
            self._start_live()
        elif target_state == States.TRAINING:
            self._ensure_training_module()
        elif target_state == States.TOOLS:
            self._ensure_tool_modules()
        elif target_state == States.PREPROCESSING:
            self._start_preprocessing()

        self.state = target_state

    def _start_live(self):
        from modules.renderloop import AsyncRenderer
        from modules.visualizer import Visualizer

        self.render_loop = AsyncRenderer()
        self.viz = Visualizer(self, self.render_loop)

        if len(self.pkls) > 0:
            for pkl in self.pkls:
                self.viz.add_recent_pickle(pkl)
            self.viz.load_pickle(self.pkls[0])

    def _ensure_training_module(self):
        if self._training_module is None:
            from modules.training_module import TrainingModule
            self._training_module = TrainingModule(ModuleHost(self))

    def _ensure_projection_module(self):
        if self._projection_module is None:
            from modules.projection_module import ProjectionModule
            self._projection_module = ProjectionModule(ModuleHost(self))

    def _ensure_features_module(self):
        if self._features_module is None:
            from modules.pca_module import PCA_Module
            self._features_module = PCA_Module(ModuleHost(self))

    def _ensure_super_res_module(self):
        if self._super_res_module is None:
            from modules.super_res_module import SuperResModule
            self._super_res_module = SuperResModule(ModuleHost(self))

    def _ensure_mixing_module(self):
        if self._mixing_module is None:
            from modules.network_mixing import MixingModule
            self._mixing_module = MixingModule(ModuleHost(self))

    def _ensure_tool_modules(self):
        self._ensure_projection_module()
        self._ensure_features_module()
        self._ensure_super_res_module()
        self._ensure_mixing_module()

    def _is_training_active(self):
        return (self._training_module is not None
                and self._training_module.is_training)

    def _start_preprocessing(self):
        from modules.preprocessing_module import DataPreprocessing
        self.data_preprocessing = DataPreprocessing(self)

    def start_renderer(self):
        self.navigate_to(States.LIVE)

    def set_visible_menu(self):
        self.navigate_to(States.LIVE)

    def start_preprocessing(self):
        self.navigate_to(States.PREPROCESSING)

    def draw_navbar(self):
        training_active = self._is_training_active()

        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.content_width, self.navbar_height)
        imgui.begin('##Navbar', closable=False, flags=(
            imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS |
            imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR))

        imgui.get_window_draw_list().add_rect_filled(
            0, 0, self.content_width, self.navbar_height,
            imgui.get_color_u32_rgba(*RED))

        logo_height = int(self.navbar_height * 0.6)
        logo_width = int(logo_height * (self.logo.shape[1] / self.logo.shape[0]))
        imgui.set_cursor_pos_y((self.navbar_height - logo_height) / 2)
        imgui.set_cursor_pos_x(20)
        imgui.image(self.logo_texture.gl_id, logo_width, logo_height)

        nav_button_height = int(self.navbar_height * 0.6)
        nav_button_y = (self.navbar_height - nav_button_height) / 2
        nav_button_width = 120

        nav_items = [
            ("Prepare", States.PREPROCESSING),
            ("Train", States.TRAINING),
            ("Perform", States.LIVE),
            ("Tools", States.TOOLS),
        ]

        imgui.same_line(20 + logo_width + 30)

        for i, (label, target_state) in enumerate(nav_items):
            if i > 0:
                imgui.same_line()

            imgui.set_cursor_pos_y(nav_button_y)
            is_active = (self.state == target_state)
            nav_disabled = training_active and target_state != States.TRAINING

            if nav_disabled:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.0, 0.0, 0.0, 0.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.0, 0.0, 0.0, 0.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.0, 0.0, 0.0, 0.0)
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0, 0.3)
            elif is_active:
                imgui.push_style_color(imgui.COLOR_BUTTON, 1.0, 1.0, 1.0, 0.3)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 1.0, 1.0, 1.0, 0.4)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 1.0, 1.0, 1.0, 0.5)
            else:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.0, 0.0, 0.0, 0.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 1.0, 1.0, 1.0, 0.15)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 1.0, 1.0, 1.0, 0.25)

            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, nav_button_height * 0.2))
            if imgui.button(label, width=nav_button_width) and not nav_disabled:
                self.navigate_to(target_state)
            imgui.pop_style_var()
            imgui.pop_style_color(4 if nav_disabled else 3)

        metacreation_height = logo_height
        metacreation_width = int(metacreation_height * (self.metacreation.shape[1] / self.metacreation.shape[0]))

        doc_button_width = 160
        doc_button_height = int(self.navbar_height * 0.6)
        doc_button_y = (self.navbar_height - doc_button_height) / 2

        imgui.same_line(self.content_width - (metacreation_width + doc_button_width + 40))
        imgui.set_cursor_pos_y(doc_button_y)

        imgui.push_style_color(imgui.COLOR_BUTTON, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 1.0, 1.0, 1.0, 0.15)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 1.0, 1.0, 1.0, 0.25)
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, doc_button_height * 0.2))
        if imgui.button("Documentation", width=doc_button_width):
            webbrowser.open(DOCS_BASE_URL)
        imgui.pop_style_var()
        imgui.pop_style_color(3)

        imgui.same_line(self.content_width - (metacreation_width + 20))
        imgui.set_cursor_pos_y((self.navbar_height - metacreation_height) / 2)
        imgui.image(self.metacreation_texture.gl_id, metacreation_width, metacreation_height)

        imgui.end()

    def _draw_module_fullscreen(self, title, module_callable):
        fullscreen_flags = (
            imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS | imgui.WINDOW_NO_TITLE_BAR)
        imgui.set_next_window_position(0, self.navbar_height)
        imgui.set_next_window_size(self.content_width, self.content_height - self.navbar_height)
        imgui.begin(title, closable=False, flags=fullscreen_flags)
        module_callable()
        imgui.end()

    def _draw_tools_grid(self):
        grid_flags = (
            imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS | imgui.WINDOW_NO_COLLAPSE)
        half_w = self.content_width / 2
        half_h = (self.content_height - self.navbar_height) / 2

        cells = [
            (0, 0, 'Project', self._projection_module),
            (1, 0, 'Extract', self._features_module),
            (0, 1, 'Upscale', self._super_res_module),
            (1, 1, 'Mix', self._mixing_module),
        ]

        for col, row, title, module in cells:
            x = col * half_w
            y = self.navbar_height + row * half_h
            imgui.set_next_window_position(x, y)
            imgui.set_next_window_size(half_w, half_h)
            imgui.begin(f'{title}##tools_grid', closable=False, flags=grid_flags)
            module()
            imgui.end()

    def draw_frame(self):

        if self.state == States.SPLASH:
            self.set_window_size(self.splash_texture.width // 2, self.splash_texture.height // 2)
            self.hide_title_bar()

        self.begin_frame()
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 4.5)

        if self.state == States.SPLASH:
            imgui.set_next_window_position(0, 0)
            imgui.set_next_window_size(self.content_width, self.content_height)
            imgui.begin('##welcome', closable=False,
                        flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR))
            imgui.image(self.splash_texture.gl_id, self.content_width, self.content_height)
            imgui.end()
            self.splash_delay -= 1
            if self.splash_delay <= 0:
                self.navigate_to(States.LIVE)
                self.set_window_size(3840, 2160)
                self.show_title_bar()

        if self.state == States.WELCOME:
            imgui.set_next_window_position(0, 0)
            imgui.set_next_window_size(self.content_width, self.content_height)
            imgui.begin('##welcome', closable=False,
                        flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR))
            imgui.image(self.splash_texture.gl_id, self.content_width, self.content_height)
            imgui.end()
            self.state = States.SPLASH
            self.splash_delay = 30

        if self.state not in (States.WELCOME, States.SPLASH):
            self.draw_navbar()

        if self.state == States.LIVE:
            if self.viz is None or self.render_loop is None:
                self.state = States.ERROR
            else:
                self.viz()

        elif self.state == States.TRAINING:
            if self._training_module is None:
                self.state = States.ERROR
            else:
                self._draw_module_fullscreen('Training##fullscreen', self._training_module)

        elif self.state == States.TOOLS:
            if (self._projection_module is None or self._features_module is None
                    or self._super_res_module is None or self._mixing_module is None):
                self.state = States.ERROR
            else:
                self._draw_tools_grid()

        elif self.state == States.PREPROCESSING:
            if self.data_preprocessing is None:
                self.state = States.ERROR
            else:
                self.data_preprocessing()

        self._adjust_font_size()
        self.end_frame()

