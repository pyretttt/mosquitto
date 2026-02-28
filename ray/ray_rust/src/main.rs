use std::path::Path;
use std::env;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

#[derive(Default)]
struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attributes = WindowAttributes::default()
                .with_title(ray_rust::project_name())
                .with_resizable(true);
            let window = event_loop
                .create_window(attributes)
                .expect("failed to create window");
            self.window = Some(window);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(window) = &self.window {
            if window.id() != window_id {
                return;
            }
        }

        if let WindowEvent::CloseRequested = event {
            event_loop.exit();
        }
    }
}

fn main() {
    let model_path = Path::new("/Users/bob/mosquitto/ray/ray_rust/resources/girl/scene.gltf");
    match ray_rust::load_gltf_summary(model_path) {
        Ok(summary) => {
            println!(
                "scenes={} nodes={} meshes={} primitives={} materials={} animations={}",
                summary.scenes,
                summary.nodes,
                summary.meshes,
                summary.primitives,
                summary.materials,
                summary.animations
            );
        }
        Err(err) => {
            eprintln!("Failed to load GLTF from path: {err}");
            let path = env::current_dir();
            println!("The current directory is {}", path.expect("Reason").display());

        },
    }

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("event loop failure");
}
