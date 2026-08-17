"""Runner to create and save a NURBS example plot."""
from pathlib import Path
from nozzlex.inverse_design.nurbs_creator import create_and_plot

out = Path(__file__).parent / 'out' / 'nurbs_example.png'
out.parent.mkdir(exist_ok=True)

# Start, End, A, B, C control points
create_and_plot((0.0, 10.0), (1.0, 1.0), (0.2, 10.0), (0.3, 5.5), (0.4, 1),
                weights=[1.0, 1.0, 1.0, 1.0, 1.0], save=str(out),
                xlim=(0.0, 1.0), ylim=(0.0, 11.0))
print('Saved:', out)
