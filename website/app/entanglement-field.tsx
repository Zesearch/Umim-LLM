"use client";

import { useEffect, useRef } from "react";

type Particle = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  phase: number;
  tone: number;
  side: -1 | 1;
};

const TAU = Math.PI * 2;

export function EntanglementField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    const pointer = { x: -1000, y: -1000, active: false };
    let particles: Particle[] = [];
    let width = 0;
    let height = 0;
    let frame = 0;

    const palette = [
      [76, 91, 214],
      [112, 80, 190],
      [47, 154, 168],
      [41, 48, 61],
    ];

    const seedParticles = () => {
      const count = width < 720 ? 44 : Math.min(92, Math.floor(width / 15));
      particles = Array.from({ length: count }, (_, index) => {
        const side: -1 | 1 = index % 2 === 0 ? -1 : 1;
        const centerX = side === -1 ? width * 0.18 : width * 0.82;
        const spreadX = width < 720 ? width * 0.35 : width * 0.24;
        return {
          x: centerX + (Math.random() - 0.5) * spreadX,
          y: height * (0.2 + Math.random() * 0.62),
          vx: (Math.random() - 0.5) * 0.13,
          vy: (Math.random() - 0.5) * 0.13,
          radius: 0.7 + Math.random() * 1.5,
          phase: Math.random() * TAU,
          tone: index % palette.length,
          side,
        };
      });
    };

    const resize = () => {
      const density = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.floor(width * density);
      canvas.height = Math.floor(height * density);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(density, 0, 0, density, 0, 0);
      seedParticles();
    };

    const drawEntanglement = (time: number) => {
      const pairs = Math.min(7, Math.floor(particles.length / 2));
      context.save();
      context.lineWidth = 0.65;

      for (let index = 0; index < pairs; index += 1) {
        const left = particles[index * 2];
        const right = particles[index * 2 + 1];
        if (!left || !right) continue;

        const pulse = 0.06 + (Math.sin(time * 0.0008 + left.phase) + 1) * 0.025;
        const gradient = context.createLinearGradient(left.x, left.y, right.x, right.y);
        gradient.addColorStop(0, `rgba(76, 91, 214, ${pulse})`);
        gradient.addColorStop(0.5, "rgba(120, 106, 190, 0.025)");
        gradient.addColorStop(1, `rgba(47, 154, 168, ${pulse})`);
        context.strokeStyle = gradient;
        context.beginPath();
        context.moveTo(left.x, left.y);
        context.bezierCurveTo(
          width * 0.38,
          left.y + Math.sin(time * 0.00045 + index) * 22,
          width * 0.62,
          right.y - Math.sin(time * 0.00045 + index) * 22,
          right.x,
          right.y,
        );
        context.stroke();
      }

      context.restore();
    };

    const render = (time = 0) => {
      context.clearRect(0, 0, width, height);

      for (let first = 0; first < particles.length; first += 1) {
        const a = particles[first];
        for (let second = first + 1; second < particles.length; second += 1) {
          const b = particles[second];
          if (a.side !== b.side) continue;
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const distance = Math.hypot(dx, dy);
          if (distance > 92) continue;

          context.strokeStyle = `rgba(55, 65, 84, ${0.055 * (1 - distance / 92)})`;
          context.lineWidth = 0.55;
          context.beginPath();
          context.moveTo(a.x, a.y);
          context.lineTo(b.x, b.y);
          context.stroke();
        }
      }

      drawEntanglement(time);

      particles.forEach((particle) => {
        if (!reduceMotion) {
          const drift = Math.sin(time * 0.00035 + particle.phase) * 0.035;
          particle.x += particle.vx + drift;
          particle.y += particle.vy + Math.cos(time * 0.0003 + particle.phase) * 0.025;

          if (pointer.active) {
            const dx = particle.x - pointer.x;
            const dy = particle.y - pointer.y;
            const distance = Math.max(24, Math.hypot(dx, dy));
            if (distance < 145) {
              particle.x += (dx / distance) * (145 - distance) * 0.006;
              particle.y += (dy / distance) * (145 - distance) * 0.006;
            }
          }

          const margin = 34;
          if (particle.x < -margin) particle.x = width + margin;
          if (particle.x > width + margin) particle.x = -margin;
          if (particle.y < -margin) particle.y = height + margin;
          if (particle.y > height + margin) particle.y = -margin;
        }

        const [red, green, blue] = palette[particle.tone];
        const glow = 0.58 + Math.sin(time * 0.001 + particle.phase) * 0.16;
        context.fillStyle = `rgba(${red}, ${green}, ${blue}, ${glow})`;
        context.beginPath();
        context.arc(particle.x, particle.y, particle.radius, 0, TAU);
        context.fill();

        if (particle.radius > 1.55) {
          context.strokeStyle = `rgba(${red}, ${green}, ${blue}, 0.1)`;
          context.lineWidth = 0.7;
          context.beginPath();
          context.arc(
            particle.x,
            particle.y,
            particle.radius * 3.8,
            time * 0.0003 + particle.phase,
            time * 0.0003 + particle.phase + Math.PI * 1.25,
          );
          context.stroke();
        }
      });

      if (!reduceMotion) frame = window.requestAnimationFrame(render);
    };

    const handlePointerMove = (event: PointerEvent) => {
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      pointer.active = true;
    };

    const handlePointerLeave = () => {
      pointer.active = false;
    };

    resize();
    render();

    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", handlePointerMove, { passive: true });
    document.addEventListener("mouseleave", handlePointerLeave);

    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", handlePointerMove);
      document.removeEventListener("mouseleave", handlePointerLeave);
    };
  }, []);

  return <canvas ref={canvasRef} className="entanglement-field" aria-hidden="true" />;
}
