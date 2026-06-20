"""Genesis-native deployment for the Wuji-hand reorientation policy.

Ports the deployment pipeline from ``wuji-mjlab/deploy/reorient`` onto the GeneLab
(Genesis) stack: real2sim cube-pose reproduction in sim and ONNX-policy control of
the hand. The pure-numpy core (frame transforms, ZMQ bridge, obs assembly, ONNX
wrapper, hand-driver abstraction) is simulator- and hardware-agnostic so it runs
and tests headlessly; Genesis viewers and real hardware live in ``deploy.scripts``.
"""
