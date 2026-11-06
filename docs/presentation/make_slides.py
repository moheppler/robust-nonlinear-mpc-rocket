from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def set_bullets(tf, items, size=18):
    tf.word_wrap = True
    first = True
    for text, level in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = text
        p.level = level
        p.space_after = Pt(6)
        for r in p.runs:
            r.font.size = Pt(size if level == 0 else size - 3)


# ---------------- Slide 1: title ----------------
s1 = prs.slides.add_slide(prs.slide_layouts[0])
s1.shapes.title.text = "SLS-based Robust MPC for the Electric Rocket"
s1.placeholders[1].text = (
    "Semester project - status update (12 min)\n"
    "The idea of system level synthesis (SLS) for robust MPC and its advantages"
)
s1.notes_slide.notes_text_frame.text = (
    "Timing: ~30 s.\n"
    "We are implementing SLS-based robust MPC (Leeman, Koehler, Zeilinger, 'Guaranteed Robust "
    "Nonlinear MPC via Disturbance Feedback') for our electric-rocket landing problem.\n"
    "Today: (1) the idea of SLS for robust MPC and why it is advantageous, (2) how it maps onto our "
    "rocket model, (3) where we are in the project and next steps."
)

# ---------------- Slide 2: the problem ----------------
s2 = prs.slides.add_slide(prs.slide_layouts[1])
s2.shapes.title.text = "The problem: constraint satisfaction under disturbances"
set_bullets(s2.placeholders[1].text_frame, [
    ("Rocket landing must respect hard constraints at all times:", 0),
    ("landing zone, velocity limits, thrust / gimbal-angle limits  ->  (x(t), u(t)) in C", 1),
    ("Uncertain dynamics:  x(t+1) = f(x(t), u(t)) + E(x(t), u(t)) * w(t),   ||w||_inf <= 1", 0),
    ("wind gusts, model mismatch, actuator dynamics", 1),
    ("Constraints must hold for EVERY disturbance realization w", 0),
    ("Nominal MPC (w = 0) violates constraints in practice;", 0),
    ("ad-hoc safety margins are safe but very conservative", 1),
], size=20)
s2.notes_slide.notes_text_frame.text = (
    "Timing: ~1 min.\n"
    "The rocket must land inside a safe zone while respecting limits on velocity, thrust and gimbal "
    "angle: (x,u) in C, a hard polytopic constraint set. The dynamics are uncertain: "
    "x+ = f(x,u) + E(x,u)w with ||w||_inf <= 1 (wind, model mismatch, actuator dynamics). The "
    "constraints must hold for every realization of w. Nominal MPC ignores w and violates the "
    "constraints; fixed ad-hoc margins are safe but overly conservative. We need a principled "
    "robust formulation that is safe by design."
)

# ---------------- Slide 3: tube MPC -> SLS ----------------
s3 = prs.slides.add_slide(prs.slide_layouts[1])
s3.shapes.title.text = "Tube MPC -> SLS: optimize the tube and the feedback jointly"
set_bullets(s3.placeholders[1].text_frame, [
    ("Tube MPC: plan a nominal trajectory (z, v) and keep an uncertainty tube around it", 0),
    ("tighten constraints by the tube support function:", 1),
    ("c^T (z_k + psi_k) + sum_j ||c^T Phi_{k,j}||_1 + b <= 0", 1),
    ("classical tube MPC fixes the ancillary feedback offline -> conservative tubes", 0),
    ("SLS / disturbance feedback: policy affine in the disturbances,", 0),
    ("u_k = v_k + psi_k^u + sum_{j<k} K_{k,j} (Delta x_j - psi_j^x)", 1),
    ("optimize the closed-loop responses Phi directly:  [I - ZA - ZBK] Phi = Sigma", 0),
    ("then x = z + psi + Phi w EXACTLY: for LTV systems the predicted tube is the true", 0),
    ("closed-loop reachable set (no compounding over-approximation); QP subproblems", 1),
], size=18)
# simple block diagram at the bottom
y = Inches(5.7)
b1 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), y, Inches(2.9), Inches(1.0))
b1.text_frame.text = "disturbance w in B"
a1 = s3.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(3.5), y + Inches(0.3), Inches(0.8), Inches(0.4))
b2 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.4), y, Inches(4.2), Inches(1.0))
b2.text_frame.text = "SLS system response Phi (decision variable)"
a2 = s3.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(8.7), y + Inches(0.3), Inches(0.8), Inches(0.4))
b3 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.6), y, Inches(3.4), Inches(1.0))
b3.text_frame.text = "x = z + psi + Phi w\n(exact tube)"
for b in (b1, b2, b3):
    for p in b.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(14)
s3.notes_slide.notes_text_frame.text = (
    "Timing: ~1.5 min.\n"
    "Tube MPC: plan a nominal trajectory (z,v) and keep a tube around it; enforce constraints by "
    "tightening with the tube's support function, so the WHOLE tube stays inside C. In classical tube "
    "MPC the feedback that keeps the error inside the tube is fixed offline, which is conservative.\n"
    "SLS / disturbance feedback instead optimizes the closed-loop response: the policy is affine in "
    "the disturbances, u_k = v_k + psi_k^u + sum_j K_{k,j}(Dx_j - psi_j^x), and we optimize the "
    "response matrices Phi directly under [I - ZA - ZBK] Phi = Sigma. Then x = z + psi + Phi w holds "
    "exactly: for LTV systems the predicted tube IS the true closed-loop reachable set (no compounding "
    "over-approximation), and each MPC subproblem is a convex QP -> real-time capable."
)

# ---------------- Slide 4: advantages ----------------
s4 = prs.slides.add_slide(prs.slide_layouts[1])
s4.shapes.title.text = "Why the SLS formulation fits our project"
set_bullets(s4.placeholders[1].text_frame, [
    ("Less conservative: feedback K, tube Phi and trajectory (z, v) optimized JOINTLY", 0),
    ("no system-specific offline design (vs fixed-feedback tube MPC)", 1),
    ("Nonlinear dynamics via SCP + dynamic linearization-error overbound:", 0),
    ("sigma_i = tau^2 mu_i + ||e_i^T E(z,v)||_1, with tau a decision variable", 1),
    ("Formal guarantees (Leeman, Koehler, Zeilinger): robust constraint satisfaction,", 0),
    ("recursive feasibility (Thm. 1), input-to-state stability (Thm. 2)", 1),
    ("Real-time: ~19.7 ms per iteration on the paper's rocket-landing example", 0),
    ("(17 states, 4 inputs, steerable thrust incl. actuator dynamics)", 1),
    ("Open-source implementation (github.com/antoineleeman/robust-nonlinear-mpc)", 0),
    ("we are adapting it to our electric-rocket model", 1),
], size=19)
s4.notes_slide.notes_text_frame.text = (
    "Timing: ~1 min.\n"
    "Advantages of the SLS formulation: (1) the feedback gains, the tube and the nominal trajectory "
    "are optimized jointly, so the tubes are much tighter than with a fixed ancillary controller and no "
    "offline system-specific design is needed; (2) nonlinearities are handled by SCP with a dynamic "
    "overbound sigma = tau^2 mu + ||e_i^T E(z,v)||_1 on the linearization + noise error, with tau "
    "itself optimized; (3) we get hard guarantees: robust constraint satisfaction, recursive feasibility "
    "and ISS; (4) it is fast - about 20 ms per MPC iteration on the rocket-landing benchmark. We build "
    "on the open-source code and adapt it to our electric rocket. Transition: 'Next: where we are...'"
)

import os
os.makedirs("/workspace/slides", exist_ok=True)
out = "/workspace/slides/sls_rmpc_intro.pptx"
prs.save(out)

# verify
prs2 = Presentation(out)
print("saved", out, "slides:", len(prs2.slides))
for i, s in enumerate(prs2.slides, 1):
    print(i, "|", s.shapes.title.text if s.shapes.title else "(no title)")
