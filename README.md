# Vulkan-Hybrid-PBR-Renderer

## Quick Links

[Milestone 1 Slides](https://docs.google.com/presentation/d/1LJ-EBeLDXqcZ0LzwrWjJPPy0J8T_foHZaoHBapNmU7w/edit?usp=sharing)

[Alpah Version](https://github.com/460xlin/Vulkan_Hybrid_PBR)

## Overview
We’d like to implement a next generation hybrid ray-raster render engine for games in Vulkan.

Based on a Vulkan render engine, we’d like to achieve the following goals:

- Implement ray-tracing features
    - Shadow
    - Transparency
    - AO (Maybe)
- Support complex scene rendering by adding GPU instancing (single mesh and maybe baked animation)
- Exploit multithreading advantage brought by Vulkan: 
    - material loading
    - LOD
    - culling
- Simple PBR

We will refer some ideas from previous 565 projects (see credits) but will implement our own framework.

## Schedule

### Monday 11/19 Milestone 1

- Full Deferred-Renderer with Albedo, Normal, Position Information and point light
- Ray-tracer with compute shader and blinn-phong shading
- Research on possibility of ray-tracing based AO

### Monday 11/26 & Wednesday 11/28  Milestone 2

- Integration
- Shadows and transparency
- BVH tree for faster ray-tracing queries

### Monday 12/03 Milestone 3
- GPU instancing
- Maybe performance analysis.
- PBR shader

### Friday 12/07 - Final Presentation

- Performance analysis
- Presentation Slides

## Credits

- [HybridRenderer](https://github.com/davidgrosman/FinalProject-HybridRenderer)
- [tinyobjloader](https://github.com/syoyo/tinyobjloader)
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [SaschaWillems/Vulkan](https://github.com/SaschaWillems/Vulkan)
