use crate::colorease::ColorEaseUniform;
use crate::termwindow::webgpu::{PostProcessUniform, ShaderUniform};
use crate::termwindow::RenderFrame;
use crate::uniforms::UniformBuilder;
use ::window::glium;
use ::window::glium::uniforms::{
    MagnifySamplerFilter, MinifySamplerFilter, Sampler, SamplerWrapFunction,
};
use ::window::glium::{BlendingFunction, LinearBlendingFactor, Surface};
use config::FreeTypeLoadTarget;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadSource {
    Intermediate,
    PingPong,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WriteTarget {
    PingPong,
    Intermediate,
    Surface,
}

pub(crate) fn ping_pong_targets(index: usize, count: usize) -> (ReadSource, WriteTarget) {
    debug_assert!(count > 0, "ping_pong_targets: count must be > 0");
    let is_last = index == count - 1;
    let read = if index % 2 == 0 {
        ReadSource::Intermediate
    } else {
        ReadSource::PingPong
    };
    let write = if is_last {
        WriteTarget::Surface
    } else if index % 2 == 0 {
        WriteTarget::PingPong
    } else {
        WriteTarget::Intermediate
    };
    (read, write)
}

impl crate::TermWindow {
    pub fn call_draw(&mut self, frame: &mut RenderFrame) -> anyhow::Result<()> {
        match frame {
            RenderFrame::Glium(ref mut frame) => self.call_draw_glium(frame),
            RenderFrame::WebGpu => self.call_draw_webgpu(),
        }
    }

    fn call_draw_webgpu(&mut self) -> anyhow::Result<()> {
        use crate::termwindow::webgpu::WebGpuTexture;

        let webgpu = self.webgpu.as_ref().unwrap();
        let render_state = self.render_state.as_ref().unwrap();

        let output = webgpu.surface.get_current_texture()?;
        let surface_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let has_postprocess = self.post_process.is_some();
        let has_bg_postprocess = self.background_post_process.is_some();

        // "main_render_target": where composited content goes before final post-process
        let main_render_target = if has_postprocess {
            &self.post_process.as_ref().unwrap().intermediate_view
        } else {
            &surface_view
        };

        let mut encoder = webgpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        let tex = render_state.glyph_cache.borrow().atlas.texture();
        let tex = tex.downcast_ref::<WebGpuTexture>().unwrap();
        let texture_view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_linear_bind_group =
            webgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &webgpu.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&webgpu.texture_linear_sampler),
                    },
                ],
                label: Some("linear bind group"),
            });

        let texture_nearest_bind_group =
            webgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &webgpu.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&webgpu.texture_nearest_sampler),
                    },
                ],
                label: Some("nearest bind group"),
            });

        let foreground_text_hsb = self.config.foreground_text_hsb;
        let foreground_text_hsb = [
            foreground_text_hsb.hue,
            foreground_text_hsb.saturation,
            foreground_text_hsb.brightness,
        ];

        let milliseconds = self.created.elapsed().as_millis() as u32;
        let projection = euclid::Transform3D::<f32, f32, f32>::ortho(
            -(self.dimensions.pixel_width as f32) / 2.0,
            self.dimensions.pixel_width as f32 / 2.0,
            self.dimensions.pixel_height as f32 / 2.0,
            -(self.dimensions.pixel_height as f32) / 2.0,
            -1.0,
            1.0,
        )
        .to_arrays_transposed();

        // Compute cursor pixel position for post-process shaders
        let cell_w = self.render_metrics.cell_size.width as f32;
        let cell_h = self.render_metrics.cell_size.height as f32;
        let (padding_left, padding_top) = self.padding_left_top();
        let tab_bar_height = if self.show_tab_bar && !self.config.tab_bar_at_bottom {
            self.tab_bar_pixel_height().unwrap_or(0.0)
        } else {
            0.0
        };
        let (cursor_pos, cursor_prev_pos, cursor_move_time) =
            if let Some(pos) = self.get_panes_to_render().into_iter().find(|p| p.is_active) {
                let cursor = pos.pane.get_cursor_position();
                let prev = self.prev_cursor.prev_position();
                let top = pos.pane.get_dimensions().physical_top;
                let to_pixel = |x: usize, y: isize| -> [f32; 2] {
                    [
                        ((x + pos.left) as f32) * cell_w + padding_left,
                        ((y + pos.top as isize - top).max(0) as f32) * cell_h
                            + tab_bar_height
                            + padding_top,
                    ]
                };
                (
                    to_pixel(cursor.x, cursor.y),
                    to_pixel(prev.x, prev.y),
                    self.prev_cursor
                        .last_cursor_movement()
                        .elapsed()
                        .as_secs_f32(),
                )
            } else {
                ([0.0f32, 0.0], [0.0f32, 0.0], 0.0)
            };

        // Build the shared PostProcessUniform
        let now = Instant::now();
        let time = self.created.elapsed().as_secs_f32();
        let time_delta = self
            .last_post_process_time
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(0.0);

        if has_postprocess || has_bg_postprocess {
            self.last_post_process_time = Some(now);
            self.post_process_frame = self.post_process_frame.wrapping_add(1);
        }

        let pp_uniform = PostProcessUniform {
            resolution: [
                self.dimensions.pixel_width as f32,
                self.dimensions.pixel_height as f32,
            ],
            time,
            time_delta,
            frame: self.post_process_frame,
            cell_width: cell_w,
            cursor_pos,
            cursor_prev_pos,
            cursor_move_time,
            cell_height: cell_h,
        };

        // Helper closure: draw a single vertex buffer to a target view
        let draw_vb = |encoder: &mut wgpu::CommandEncoder,
                       vb: &crate::renderstate::TripleVertexBuffer,
                       target: &wgpu::TextureView,
                       cleared: &mut bool| {
            let (vertex_count, index_count) = vb.vertex_index_count();
            if vertex_count > 0 {
                let mut vertices = vb.current_vb_mut();
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: if *cleared {
                                wgpu::LoadOp::Load
                            } else {
                                wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.,
                                    g: 0.,
                                    b: 0.,
                                    a: 0.,
                                })
                            },
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                *cleared = true;

                let uniforms = webgpu.create_uniform(ShaderUniform {
                    foreground_text_hsb,
                    milliseconds,
                    projection,
                });

                render_pass.set_pipeline(&webgpu.render_pipeline);
                render_pass.set_bind_group(0, &uniforms, &[]);
                render_pass.set_bind_group(1, &texture_linear_bind_group, &[]);
                render_pass.set_bind_group(2, &texture_nearest_bind_group, &[]);
                let vertex_buffer = vertices.webgpu_mut().recreate();
                vertex_buffer.unmap();
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(vb.indices.webgpu().slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..index_count as _, 0, 0..1);
            }
            vb.next_index();
        };

        if has_bg_postprocess {
            // Phase-based rendering: background → bg_pp → text/cursor → final pp
            let bg_target = &self.background_post_process.as_ref().unwrap().intermediate_view;

            // Phase 1: Draw all idx==0 (backgrounds) to bg_pp intermediate
            let mut bg_cleared = false;
            {
                let layers = render_state.layers.borrow();
                for layer in layers.iter() {
                    let vbs = layer.vb.borrow();
                    draw_vb(&mut encoder, &vbs[0], bg_target, &mut bg_cleared);
                    // Advance idx 1,2 without drawing (they'll be drawn in Phase 3)
                    vbs[1].next_index();
                    vbs[2].next_index();
                }
            }

            // Phase 2: Run background post-process shader chain
            // Reads bg_pp.intermediate, last shader writes to main_render_target
            if let Some(bg_pp) = &self.background_post_process {
                webgpu
                    .queue
                    .write_buffer(&bg_pp.uniform_buffer, 0, bytemuck::cast_slice(&[pp_uniform]));

                let num = bg_pp.pipelines.len();
                for (i, pipeline) in bg_pp.pipelines.iter().enumerate() {
                    let is_last = i == num - 1;

                    let read_bind_group = if i % 2 == 0 {
                        &bg_pp.bind_group_intermediate
                    } else {
                        bg_pp.bind_group_pingpong.as_ref().unwrap()
                    };

                    let write_view = if is_last {
                        main_render_target
                    } else if i % 2 == 0 {
                        bg_pp.ping_pong_view.as_ref().unwrap()
                    } else {
                        &bg_pp.intermediate_view
                    };

                    let mut render_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Background PostProcess Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: write_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 0.,
                                        g: 0.,
                                        b: 0.,
                                        a: 0.,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        });

                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(0, read_bind_group, &[]);
                    render_pass.set_bind_group(1, &bg_pp.uniform_bind_group, &[]);
                    render_pass.set_bind_group(2, &bg_pp.bind_group_prev_frame, &[]);
                    render_pass.draw(0..3, 0..1);
                }
            }

            // Copy bg_pp result to prev_frame for feedback BEFORE text is drawn
            // main_render_target now has the post-processed background
            if let Some(bg_pp) = &self.background_post_process {
                if has_postprocess {
                    // main_render_target is pp.intermediate which has COPY_SRC
                    let pp = self.post_process.as_ref().unwrap();
                    encoder.copy_texture_to_texture(
                        pp.intermediate_texture.as_image_copy(),
                        bg_pp.prev_frame_texture.as_image_copy(),
                        pp.intermediate_texture.size(),
                    );
                }
            }

            // Phase 3: Draw all idx==1,2 (text, cursor) on top of main_render_target
            let mut main_cleared = true; // bg_pp already wrote to it
            {
                let layers = render_state.layers.borrow();
                for layer in layers.iter() {
                    let vbs = layer.vb.borrow();
                    // idx 0 already advanced in Phase 1
                    draw_vb(&mut encoder, &vbs[1], main_render_target, &mut main_cleared);
                    draw_vb(&mut encoder, &vbs[2], main_render_target, &mut main_cleared);
                }
            }
        } else {
            // Original single-pass rendering (no background post-process)
            let mut cleared = false;
            let layers = render_state.layers.borrow();
            for layer in layers.iter() {
                let vbs = layer.vb.borrow();
                for idx in 0..3 {
                    draw_vb(&mut encoder, &vbs[idx], main_render_target, &mut cleared);
                }
            }
        }

        // Run final post-process shader chain if active
        if let Some(pp) = &self.post_process {
            webgpu
                .queue
                .write_buffer(&pp.uniform_buffer, 0, bytemuck::cast_slice(&[pp_uniform]));

            let num_pipelines = pp.pipelines.len();

            for (i, pipeline) in pp.pipelines.iter().enumerate() {
                let (read_src, write_dst) = ping_pong_targets(i, num_pipelines);

                let read_bind_group = match read_src {
                    ReadSource::Intermediate => &pp.bind_group_intermediate,
                    ReadSource::PingPong => pp.bind_group_pingpong.as_ref().unwrap(),
                };

                let write_view = match write_dst {
                    WriteTarget::Surface => &surface_view,
                    WriteTarget::PingPong => pp.ping_pong_view.as_ref().unwrap(),
                    WriteTarget::Intermediate => &pp.intermediate_view,
                };

                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("PostProcess Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: write_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.,
                                g: 0.,
                                b: 0.,
                                a: 0.,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, read_bind_group, &[]);
                render_pass.set_bind_group(1, &pp.uniform_bind_group, &[]);
                render_pass.set_bind_group(2, &pp.bind_group_prev_frame, &[]);
                render_pass.draw(0..3, 0..1);
            }
        }

        // Copy pp intermediate to prev_frame for feedback (cursor trail etc.)
        // pp.intermediate holds the full rendered content (bg + text) before final shader
        if let Some(pp) = &self.post_process {
            encoder.copy_texture_to_texture(
                pp.intermediate_texture.as_image_copy(),
                pp.prev_frame_texture.as_image_copy(),
                pp.intermediate_texture.size(),
            );
        }

        // submit will accept anything that implements IntoIter
        webgpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn call_draw_glium(&mut self, frame: &mut glium::Frame) -> anyhow::Result<()> {
        use window::glium::texture::SrgbTexture2d;

        let gl_state = self.render_state.as_ref().unwrap();
        let tex = gl_state.glyph_cache.borrow().atlas.texture();
        let tex = tex.downcast_ref::<SrgbTexture2d>().unwrap();

        frame.clear_color(0., 0., 0., 0.);

        let projection = euclid::Transform3D::<f32, f32, f32>::ortho(
            -(self.dimensions.pixel_width as f32) / 2.0,
            self.dimensions.pixel_width as f32 / 2.0,
            self.dimensions.pixel_height as f32 / 2.0,
            -(self.dimensions.pixel_height as f32) / 2.0,
            -1.0,
            1.0,
        )
        .to_arrays_transposed();

        let use_subpixel = match self
            .config
            .freetype_render_target
            .unwrap_or(self.config.freetype_load_target)
        {
            FreeTypeLoadTarget::HorizontalLcd | FreeTypeLoadTarget::VerticalLcd => true,
            _ => false,
        };

        let dual_source_blending = glium::DrawParameters {
            blend: glium::Blend {
                color: BlendingFunction::Addition {
                    source: LinearBlendingFactor::SourceOneColor,
                    destination: LinearBlendingFactor::OneMinusSourceOneColor,
                },
                alpha: BlendingFunction::Addition {
                    source: LinearBlendingFactor::SourceOneColor,
                    destination: LinearBlendingFactor::OneMinusSourceOneColor,
                },
                constant_value: (0.0, 0.0, 0.0, 0.0),
            },

            ..Default::default()
        };

        let alpha_blending = glium::DrawParameters {
            blend: glium::Blend {
                color: BlendingFunction::Addition {
                    source: LinearBlendingFactor::SourceAlpha,
                    destination: LinearBlendingFactor::OneMinusSourceAlpha,
                },
                alpha: BlendingFunction::Addition {
                    source: LinearBlendingFactor::One,
                    destination: LinearBlendingFactor::OneMinusSourceAlpha,
                },
                constant_value: (0.0, 0.0, 0.0, 0.0),
            },
            ..Default::default()
        };

        // Clamp and use the nearest texel rather than interpolate.
        // This prevents things like the box cursor outlines from
        // being randomly doubled in width or height
        let atlas_nearest_sampler = Sampler::new(&*tex)
            .wrap_function(SamplerWrapFunction::Clamp)
            .magnify_filter(MagnifySamplerFilter::Nearest)
            .minify_filter(MinifySamplerFilter::Nearest);

        let atlas_linear_sampler = Sampler::new(&*tex)
            .wrap_function(SamplerWrapFunction::Clamp)
            .magnify_filter(MagnifySamplerFilter::Linear)
            .minify_filter(MinifySamplerFilter::Linear);

        let foreground_text_hsb = self.config.foreground_text_hsb;
        let foreground_text_hsb = (
            foreground_text_hsb.hue,
            foreground_text_hsb.saturation,
            foreground_text_hsb.brightness,
        );

        let milliseconds = self.created.elapsed().as_millis() as u32;

        let cursor_blink: ColorEaseUniform = (*self.cursor_blink_state.borrow()).into();
        let blink: ColorEaseUniform = (*self.blink_state.borrow()).into();
        let rapid_blink: ColorEaseUniform = (*self.rapid_blink_state.borrow()).into();

        for layer in gl_state.layers.borrow().iter() {
            for idx in 0..3 {
                let vb = &layer.vb.borrow()[idx];
                let (vertex_count, index_count) = vb.vertex_index_count();
                if vertex_count > 0 {
                    let vertices = vb.current_vb_mut();
                    let subpixel_aa = use_subpixel && idx == 1;

                    let mut uniforms = UniformBuilder::default();

                    uniforms.add("projection", &projection);
                    uniforms.add("atlas_nearest_sampler", &atlas_nearest_sampler);
                    uniforms.add("atlas_linear_sampler", &atlas_linear_sampler);
                    uniforms.add("foreground_text_hsb", &foreground_text_hsb);
                    uniforms.add("subpixel_aa", &subpixel_aa);
                    uniforms.add("milliseconds", &milliseconds);
                    uniforms.add_struct("cursor_blink", &cursor_blink);
                    uniforms.add_struct("blink", &blink);
                    uniforms.add_struct("rapid_blink", &rapid_blink);

                    frame.draw(
                        vertices.glium().slice(0..vertex_count).unwrap(),
                        vb.indices.glium().slice(0..index_count).unwrap(),
                        gl_state.glyph_prog.as_ref().unwrap(),
                        &uniforms,
                        if subpixel_aa {
                            &dual_source_blending
                        } else {
                            &alpha_blending
                        },
                    )?;
                }

                vb.next_index();
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ping_pong_single_shader() {
        let (read, write) = ping_pong_targets(0, 1);
        assert_eq!(read, ReadSource::Intermediate);
        assert_eq!(write, WriteTarget::Surface);
    }

    #[test]
    fn test_ping_pong_two_shaders() {
        let (r0, w0) = ping_pong_targets(0, 2);
        assert_eq!(r0, ReadSource::Intermediate);
        assert_eq!(w0, WriteTarget::PingPong);

        let (r1, w1) = ping_pong_targets(1, 2);
        assert_eq!(r1, ReadSource::PingPong);
        assert_eq!(w1, WriteTarget::Surface);
    }

    #[test]
    fn test_ping_pong_three_shaders() {
        let (r0, w0) = ping_pong_targets(0, 3);
        assert_eq!(r0, ReadSource::Intermediate);
        assert_eq!(w0, WriteTarget::PingPong);

        let (r1, w1) = ping_pong_targets(1, 3);
        assert_eq!(r1, ReadSource::PingPong);
        assert_eq!(w1, WriteTarget::Intermediate);

        let (r2, w2) = ping_pong_targets(2, 3);
        assert_eq!(r2, ReadSource::Intermediate);
        assert_eq!(w2, WriteTarget::Surface);
    }
}
