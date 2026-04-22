use mux::renderable::StableCursorPosition;
use std::time::Instant;

#[derive(Clone)]
pub struct PrevCursorPos {
    pos: StableCursorPosition,
    prev_pos: StableCursorPosition,
    when: Instant,
}

impl PrevCursorPos {
    pub fn new() -> Self {
        PrevCursorPos {
            pos: StableCursorPosition::default(),
            prev_pos: StableCursorPosition::default(),
            when: Instant::now(),
        }
    }

    /// Make the cursor look like it moved
    pub fn bump(&mut self) {
        self.when = Instant::now();
    }

    /// Update the cursor position if its different.
    /// Only updates prev_pos if the cursor was stationary for a while,
    /// so that rapid consecutive moves preserve the original "from" position.
    pub fn update(&mut self, newpos: &StableCursorPosition) {
        if &self.pos != newpos {
            let now = Instant::now();
            let gap = now.duration_since(self.when);
            // Only record prev_pos if cursor was still for >50ms
            if gap.as_millis() > 50 {
                self.prev_pos = self.pos;
            }
            self.pos = *newpos;
            self.when = now;
        }
    }

    /// When did the cursor last move?
    pub fn last_cursor_movement(&self) -> Instant {
        self.when
    }

    /// Return the current cursor position
    pub fn position(&self) -> &StableCursorPosition {
        &self.pos
    }

    /// Return the position before the last move
    pub fn prev_position(&self) -> &StableCursorPosition {
        &self.prev_pos
    }
}
