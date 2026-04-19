
try:
    import mediapipe as mp
    print(f"MediaPipe file: {mp.__file__}")
    print(f"Has solutions? {hasattr(mp, 'solutions')}")
    if hasattr(mp, 'solutions'):
        print(f"Solutions: {mp.solutions}")
        print(f"Face Mesh: {mp.solutions.face_mesh}")
    else:
        print("Dir of mp:", dir(mp))
except Exception as e:
    print(e)
