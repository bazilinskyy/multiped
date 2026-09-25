using System.Collections;
using System.Collections.Generic;
using System.IO; // For writing .txt logs
using UnityEngine;
using Pixelplacement;
using UnityEngine.UI;
using UnityEngine.XR;
using PlayFab;
using PlayFab.ClientModels;
using PlayFab.DataModels;
using PlayFab.ProfilesModels;
using UnityStandardAssets.ImageEffects;
using UnityEngine.SceneManagement;

/// <summary>
/// Controls car movement along predefined splines, handles yielding to pedestrians,
/// logs crossing/yield events to file/console/PlayFab, and drives eHMI light logic.
/// </summary>
public class CarMovement : MonoBehaviour
{
    // ========================= Car movement setup =========================

    /// <summary>Transform of the car that moves along splines.</summary>
    public Transform myObject;

    /// <summary>Splines for segmented routes (yield scenarios) and the full route (no yield).</summary>
    public Spline FirstSpline; public Spline SecondSpline; public Spline ThirdSpline; public Spline FourthSpline; public Spline FullSpline;

    /// <summary>Easing curve for the wheel rotation of the non-yielding car.</summary>
    public AnimationCurve FullCurve;

    /// <summary>Reference point on car route used for distance checks.</summary>
    public GameObject measuringPoint;
    /// <summary>Current distance of car along measurement axis (meters).</summary>
    public float carDistance;

    // Wheel rotation parameters
    private int Ani = 12;
    private float wheelSize = 0.5f;
    private int Dist = 145;

    /// <summary>Wheel mesh references (used for rotation animations).</summary>
    public GameObject Lfront; public GameObject Lrear; public GameObject Rfront; public GameObject Rrear;

    /// <summary>Number of cars spawned in current wave.</summary>
    public int carCount = 0;

    /// <summary>Yielding mode: 1 = yield to P1, 2 = yield to P2, 0 = no yield.</summary>
    public int Yield;

    /// <summary>Start time (Time.time) of the active car run.</summary>
    public float startTime;

    // Predefined yield arrays for different modes
    int[] yieldArrayDemo = { 1, 2, 0 };
    int[] yieldArrayPreview = { 1 };

    int[] yieldArray;
    int[] yieldArrayCondition1 = { 0 };
    int[] yieldArrayCondition2 = { 1 };
    int[] yieldArrayCondition3 = { 2 };
    int[] yieldArrayCondition4 = { 0 };
    int[] yieldArrayCondition5 = { 0 };

    /// <summary>True while the Wave coroutine is actively spawning/driving cars.</summary>
    public bool WaveStarted = false;

    // ========================= Scene object refs & state =========================

    /// <summary>Proxy object that moves with the car and is used for distance checks.</summary>
    public GameObject distance_cube;

    /// <summary>Pedestrian objects for distance/crossing checks.</summary>
    public GameObject pedestrian1;
    public GameObject pedestrian2;

    /// <summary>3D distances from car proxy to pedestrians.</summary>
    public float pedestrian1_distance;
    public float pedestrian2_distance;

    /// <summary>Absolute X-axis distances from car proxy to pedestrians.</summary>
    public float pedestrian1_distance_x;
    public float pedestrian2_distance_x;

    /// <summary>Current commanded car speed (km/h).</summary>
    public float speed;

    /// <summary>True while we are in a yielding phase (approach/stop/resume) for the current car.</summary>
    public bool yielding;

    float fixedDeltaTime;             // Elapsed time since car start (cached each FixedUpdate)

    LightStripBumper LEDscript;       // eHMI LED controller
    /// <summary>True once all cars for this condition have finished driving.</summary>
    public bool conditionFinished = false;

    ConditionController conditionScript; // Global experiment/condition controller
    PlayFabController playfabScript;     // PlayFab logger (optional)

    /// <summary>Audio cue for vehicle spawn/beep.</summary>
    public AudioSource AudioBeep;
    /// <summary>Audio cue for counts (unused in this snippet but wired).</summary>
    public AudioSource CountSound;

    // ========================= Crossing logging =========================

    [Header("Crossing logging")]
    [SerializeField]
    [Tooltip("Only log a crossing if the car is within this 3D distance (m) of the pedestrian.")]
    float crossProximityMeters = 6f;

    bool p1Crossed, p2Crossed;  // Whether crossing for each pedestrian has been logged this run
    float? lastRelX1, lastRelX2; // Previous-frame signed X difference (car.x - ped.x) for zero-crossing detection

    // ========================= File logging =========================

    string logFilePath; // Absolute path to the text log file
    bool logInitialized = false;

    // ========================= Speed limiting (non-yield) =========================

    [Header("Speed limit (non-yield)")]
    [Tooltip("Maximum physical speed the car is allowed to reach in km/h when not yielding.")]
    public float maxSpeedKmh = 50f;

    // With 40 km/h/s the non-yielding AV reaches 50 km/h 1.25 s after trial onset,
    // more than 100 m before the pedestrians, and then drives at a constant speed.
    [Tooltip("Acceleration toward max speed in km/h per second (non-yield).")]
    public float accelKmhPerSec = 40f;

    // Internal state for manual movement on FullSpline (non-yield)
    bool _fullManualActive = false;
    float _fullT = 0f;               // current param on [0,1] along FullSpline
    float _currentSpeedMps = 0f;     // current speed in m/s (for movement, not logging)

    // ========================= Yielding: constant deceleration =========================
    // A yielding AV accelerates from rest like the non-yielding AV (accelKmhPerSec), cruises at
    // yieldCruiseSpeedKmh, brakes at exactly yieldDecelerationMps2 so that it stops at the end of the
    // stop spline (FirstSpline for Yield = 1, ThirdSpline for Yield = 2), stands still for exactly
    // yieldStandstillS, and drives off along the next spline. The braking distance follows from
    // v^2 / (2a): 40.19 m for 50 km/h and 2.4 m/s^2. Event times in the log are exact (interpolated
    // within the physics step), and the logged speed is the commanded speed.
    // (The multiped experiment used a tween with a hand-drawn speed curve instead, so its braking
    // was not at a constant rate; that motion has been removed.)

    [Header("Yielding (constant deceleration)")]
    [Tooltip("Cruise speed before braking (km/h).")]
    public float yieldCruiseSpeedKmh = 50f;

    [Tooltip("Constant braking deceleration (m/s^2).")]
    public float yieldDecelerationMps2 = 2.4f;

    [Tooltip("Time at standstill before driving off (s).")]
    public float yieldStandstillS = 3f;

    [Tooltip("Acceleration when driving off after the standstill (m/s^2).")]
    public float yieldDriveOffAccelMps2 = 2f;

    enum YieldPhase { Idle, Approach, Braking, Standstill, DriveOff, Done }
    YieldPhase _yPhase = YieldPhase.Idle;

    Spline _yStopSpline, _yGoSpline;   // spline ending at the stop point, and the spline after it
    float[] _yStopLut, _yGoLut;        // cumulative arc length (m) at t = i / ArcLutSteps
    float _yStopLength, _yGoLength;    // spline lengths (m)
    float _yS;                         // arc length travelled along the current spline (m)
    float _ySpeedMps;                  // commanded speed (m/s)
    float _yBrakeOnsetS;               // arc length on the stop spline where braking begins (m)
    float _yBrakeDecelMps2;            // deceleration actually applied (equals yieldDecelerationMps2 when at cruise speed)
    float _yStandstillElapsed;         // time spent at standstill (s)
    const int ArcLutSteps = 1000;

    /// <summary>
    /// Unity Awake: cache AudioSource reference and initialize logging.
    /// </summary>
    public void Awake()
    {
        AudioBeep = GetComponent<AudioSource>();

        // Initialize logging here, BEFORE any Start() on any script
#if UNITY_EDITOR
        logFilePath = Path.Combine(Application.dataPath, "CarCrossingLog.txt");
#else
        logFilePath = Path.Combine(Application.persistentDataPath, "CarCrossingLog.txt");
#endif

        logInitialized = true;

        EnsureLogFileExists();
        AppendLogToFile("=== Car Crossing Log STARTED ===");
        Debug.Log("Logging to: " + logFilePath);
    }

    /// <summary>
    /// Unity Start: check the scene references (logging already initialized in Awake).
    /// </summary>
    void Start()
    {
        if (distance_cube == null)
            Debug.LogWarning("CarMovement: distance_cube is not assigned. Distances and crossings will not be logged until it is set.");
    }

    /// <summary>
    /// Ensures the directory and the log file exist, creating them if necessary.
    /// </summary>
    void EnsureLogFileExists()
    {
        try
        {
            var dir = Path.GetDirectoryName(logFilePath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            if (!File.Exists(logFilePath))
            {
                using (var sw = new StreamWriter(logFilePath, false))
                {
                    sw.WriteLine("=== Car Crossing Log ===");
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError("Failed to ensure log file exists at: " + logFilePath + "\n" + e);
        }
    }

    /// <summary>
    /// Appends a single message line to the log file. Swallows errors with a console log.
    /// </summary>
    /// <param name="message">Line to write (no newline required).</param>
    void AppendLogToFile(string message)
    {
        // Only care that we have a valid path
        if (string.IsNullOrEmpty(logFilePath))
        {
            Debug.LogError("CarMovement: logFilePath is null or empty, cannot write log line.");
            return;
        }

        try
        {
            using (var sw = new StreamWriter(logFilePath, true))
            {
                sw.WriteLine(message);
            }
        }
        catch (System.Exception e) // broader than IOException to catch all issues
        {
            Debug.LogError("Could not write to log file at: " + logFilePath + "\n" + e);
        }
    }

    /// <summary>
    /// Unified logger: mirrors to Console, PlayFab buffer (if available), and file.
    /// Adds eHMI state and optionally pedestrian positions.
    /// </summary>
    void LogLine(string msg, Vector3? p1 = null, Vector3? p2 = null)
    {
        string ehmi = (conditionScript != null && conditionScript.eHMIOn == 1) ? "On" : "Off";
        msg += $" | eHMI={ehmi}";
        if (p1.HasValue && p2.HasValue)
        {
            msg += $" | P1=({p1.Value.x:F2},{p1.Value.y:F2},{p1.Value.z:F2}) " +
                   $"P2=({p2.Value.x:F2},{p2.Value.y:F2},{p2.Value.z:F2})";
        }
        Debug.Log(msg);
        if (playfabScript != null) playfabScript.ButtonDataList.Add(msg);
        AppendLogToFile(msg);
    }

#if UNITY_EDITOR
    /// <summary>
    /// Context-menu helper to test file writing from the Editor.
    /// </summary>
    [ContextMenu("TEST: Append log line")]
    void TestAppendLog()
    {
        AppendLogToFile($"[TEST] {System.DateTime.Now:HH:mm:ss} — writing to {logFilePath}");
        Debug.Log("Wrote test line to: " + logFilePath);
    }
#endif

    /// <summary>
    /// Launches a demo sequence: cycles through yield modes (P1, P2, none) and logs speed.
    /// </summary>
    public void StartCarDemo()
    {
        carCount = 0;
        yieldArray = yieldArrayDemo;
        conditionScript = GameObject.Find("ConditionController").GetComponent<ConditionController>();
        StartCoroutine("Wave");
    }

    /// <summary>
    /// Launches a preview sequence: uses a single yield-to-P1 scenario and enables eHMI if active.
    /// </summary>
    public void StartCarPreview()
    {
        carCount = 0;
        yieldArray = yieldArrayPreview;
        conditionScript = GameObject.Find("ConditionController").GetComponent<ConditionController>();
        StartCoroutine("Wave");
        if (conditionScript.eHMIOn == 1)
        {
            LEDscript = GameObject.Find("LightStrip").GetComponent<LightStripBumper>();
        }
    }

    /// <summary>
    /// Launches a trial run based on the current ConditionController settings:
    /// enables/disables eHMI, sets yield target, and starts the wave.
    /// </summary>
    public void StartCar()
    {
        carCount = 0;
        conditionScript = GameObject.Find("ConditionController").GetComponent<ConditionController>();
        playfabScript = GameObject.Find("PlayFabController").GetComponent<PlayFabController>();

        string time = System.DateTime.UtcNow.AddHours(2f).ToString();
        playfabScript.ButtonDataList.Add(time); // Add time to PlayFab buffer

        yieldArray = yieldArrayCondition1; // Currently fixed; replace with condition mapping if needed

        // Trigger eHMI
        if (conditionScript.eHMIOn == 1)
        {
            Debug.Log("eHMI enabled");
            LEDscript = GameObject.Find("LightStrip").GetComponent<LightStripBumper>();
        }
        else
        {
            Debug.Log("eHMI disabled");
            LEDscript = GameObject.Find("LightStrip")?.GetComponent<LightStripBumper>();
            GameObject.Find("LightStrip").SetActive(false);
        }

        // Set yielding based on condition (1 => P1, 2 => P2, else none)
        if (conditionScript.yielding == 1)
        {
            Debug.Log("Yielding ON for P1");
            Yield = 1;
        }
        else if (conditionScript.yielding == 2)
        {
            Debug.Log("Yielding ON for P2");
            Yield = 2;
        }
        else
        {
            Debug.Log("Yielding OFF");
            Yield = 0;
        }
        StartCoroutine("Wave");
    }

    /// <summary>
    /// Unity FixedUpdate: updates movement, distances, estimates speed, detects crossings/yield events,
    /// and writes relevant logs each physics tick.
    /// </summary>
    public void FixedUpdate()
    {
        // First, advance manual non-yield motion (if active)
        if (Yield == 0 && WaveStarted)
        {
            UpdateNonYieldMotion();
        }

        // Yielding motion: constant deceleration
        if (Yield > 0 && WaveStarted)
        {
            UpdateConstantDecelYield();
        }

        // If conditionScript is not yet assigned, avoid null reference crashes
        if (conditionScript == null || distance_cube == null)
            return;

        fixedDeltaTime = Time.time - startTime;

        // Compute distances (3D and X-axis only) to each pedestrian
        pedestrian1_distance = Vector3.Distance(distance_cube.transform.position, conditionScript.p1_object.transform.position);
        pedestrian2_distance = Vector3.Distance(distance_cube.transform.position, conditionScript.p2_object.transform.position);
        pedestrian1_distance_x = Mathf.Abs(distance_cube.transform.position.x - conditionScript.p1_object.transform.position.x);
        pedestrian2_distance_x = Mathf.Abs(distance_cube.transform.position.x - conditionScript.p2_object.transform.position.x);

        // Car distance along measurement axis (3D)
        carDistance = Vector3.Distance(measuringPoint.transform.position, distance_cube.transform.position);

        // Commanded speed (km/h)
        speed = WaveStarted ? (Yield > 0 ? _ySpeedMps : _currentSpeedMps) * 3.6f : 0f;

        // Continuous per-tick console log (verbose)
        Debug.Log($"[car_t={(Time.time - startTime):F2}s] continous_pedestrian1_distance= {pedestrian1_distance_x} pedestrian2_distance= {pedestrian2_distance_x}");

        // ====================== Crossing detection (zero-crossing on X with proximity gate) ======================
        Vector3 carPos = distance_cube.transform.position;
        Vector3 p1Pos = conditionScript.p1_object.transform.position;
        Vector3 p2Pos = conditionScript.p2_object.transform.position;

        float relX1 = carPos.x - p1Pos.x;
        float relX2 = carPos.x - p2Pos.x;

        bool nearP1 = pedestrian1_distance <= crossProximityMeters;
        bool nearP2 = pedestrian2_distance <= crossProximityMeters;

        // Distance between pedestrians (for context in logs)
        float pedestrianPairDistance = Vector3.Distance(p1Pos, p2Pos);

        // P1 crossing (edge-triggered when sign of relX flips while within proximity)
        if (!p1Crossed)
        {
            if (lastRelX1.HasValue)
            {
                if (Mathf.Sign(relX1) != Mathf.Sign(lastRelX1.Value) && nearP1)
                {
                    p1Crossed = true;
                    float t = Time.time - startTime;
                    string wallClock = System.DateTime.UtcNow.AddHours(2f).ToString("HH:mm:ss");
                    string exact = $" | t_exact={ExactCrossTime(t, lastRelX1.Value, relX1):F3}s";
                    LogLine($"[CROSS] P1 at t={t:F2}s (wall {wallClock}) | carDist={carDistance:F1}m | speed={speed:F1} km/h | pedDistance={pedestrianPairDistance:F2}m" + exact,
                            p1Pos, p2Pos);
                }
            }
            lastRelX1 = relX1;
        }

        // P2 crossing
        if (!p2Crossed)
        {
            if (lastRelX2.HasValue)
            {
                if (Mathf.Sign(relX2) != Mathf.Sign(lastRelX2.Value) && nearP2)
                {
                    p2Crossed = true;
                    float t = Time.time - startTime;
                    string wallClock = System.DateTime.UtcNow.AddHours(2f).ToString("HH:mm:ss");
                    string exact = $" | t_exact={ExactCrossTime(t, lastRelX2.Value, relX2):F3}s";
                    LogLine($"[CROSS] P2 at t={t:F2}s (wall {wallClock}) | carDist={carDistance:F1}m | speed={speed:F1} km/h | pedDistance={pedestrianPairDistance:F2}m" + exact,
                            p1Pos, p2Pos);
                }
            }
            lastRelX2 = relX2;
        }
        // ========================================================================================================

        // Yield events (start/stop/resume/end) are logged with exact times by UpdateConstantDecelYield.
    }

    /// <summary>
    /// Main wave coroutine: iterates over yieldArray, resets state,
    /// drives a car per entry, and spaces spawns with audio cues.
    /// </summary>
    IEnumerator Wave()
    {
        for (; ; )
        {
            WaveStarted = true; // Enables speed updates and LED logic

            // If we haven't reached the maximum amount of cars yet
            if (carCount < yieldArray.Length)
            {
                // Reset crossing state for this new car BEFORE it starts
                p1Crossed = p2Crossed = false;
                lastRelX1 = lastRelX2 = null;

                // --- start a new trial block in the log --- //
                AppendLogToFile(""); // blank line
                string ehmi = (conditionScript != null && conditionScript.eHMIOn == 1) ? "On" : "Off";
                AppendLogToFile($"--- TRIAL #{carCount + 1} (Yield={Yield}, eHMI={ehmi}) ---");
                AppendLogToFile(""); // another blank line
                // ----------------------------------------- //

                startTime = Time.time; // Start time of car route

                if (conditionScript.trial)
                {
                    if (playfabScript != null)
                        playfabScript.ButtonDataList.Add("(" + (Yield).ToString() + ")"); // Add Yield number to playfab data
                }

                DriveCar();   // Kick off tweens or manual movement for the current yield mode
                carCount += 1;
            }
            else
            {
                // END: all cars for this wave have run
                Debug.Log("car movement finished");
                conditionFinished = true;
                StopCoroutine("Wave");
            }

            // Delay until next vehicle starts (longer when yielding)
            if (Yield > 0)
            {
                yield return new WaitForSecondsRealtime(19f);
                if (carCount < yieldArray.Length)
                {
                    AudioBeep.Play();
                    yield return new WaitForSecondsRealtime(1f);
                }
            }
            else
            {
                yield return new WaitForSecondsRealtime(12f);
                if (carCount < yieldArray.Length)
                {
                    AudioBeep.Play();
                    yield return new WaitForSecondsRealtime(1f);
                }
            }

            WaveStarted = false;

            // Reset light strip counters between cars (when relevant)
            if (conditionScript.conditionCounter > 1 && LEDscript != null)
            {
                LEDscript.counter = 0;
                LEDscript.counter2 = 0;
            }
        }
    }

    /// <summary>
    /// Starts spline tweens and wheel rotations for the current Yield mode.
    /// Non-yield case uses manual movement with a physical speed cap.
    /// </summary>
    public void DriveCar()
    {
        // Align trial time with the physics clock, so that the first motion step covers
        // t = 0 to t = fixedDeltaTime and logged event times are exact.
        startTime = Time.fixedTime;

        if (Yield > 0)
        {
            StartConstantDecelYield();
            return;
        }

        if (Yield == 0)
        {
            // --- Manual movement along FullSpline with speed cap ---

            // Reset manual movement state
            _fullManualActive = true;
            _fullT = 0f;
            _currentSpeedMps = 0f;

            if (FullSpline == null || myObject == null)
            {
                Debug.LogError("CarMovement: FullSpline or myObject not assigned, cannot drive non-yield manually.");
                _fullManualActive = false;
            }
            else
            {
                // Snap car to start of spline
                myObject.position = FullSpline.GetPosition(0f);
                OrientAlongFullSpline(0f);
            }

            // Keep wheel rotations driven by LeanTween as before
            WheelSpin0 LF = new WheelSpin0(Lfront, FullCurve, Dist, wheelSize); LF.SetupTween(Ani, 0);
            WheelSpin0 LR = new WheelSpin0(Lrear, FullCurve, Dist, wheelSize); LR.SetupTween(Ani, 0);
            WheelSpin0 RF = new WheelSpin0(Rfront, FullCurve, Dist, wheelSize); RF.SetupTween(Ani, 0);
            WheelSpin0 RR = new WheelSpin0(Rrear, FullCurve, Dist, wheelSize); RR.SetupTween(Ani, 0);

            // IMPORTANT: do NOT call Tween.Spline for FullSpline here anymore.
        }
    }

    /// <summary>
    /// Manually advances the car along FullSpline in the non-yield case,
    /// enforcing a physical speed cap in world space.
    /// </summary>
    void UpdateNonYieldMotion()
    {
        if (!_fullManualActive || FullSpline == null || myObject == null) return;

        float dt = Time.fixedDeltaTime;

        // Convert config values to m/s and m/s^2
        float maxSpeedMps = maxSpeedKmh / 3.6f;
        float accelMps2   = accelKmhPerSec / 3.6f;

        // Accelerate toward max speed, but never exceed it
        _currentSpeedMps = Mathf.MoveTowards(_currentSpeedMps, maxSpeedMps, accelMps2 * dt);

        // Maximum distance we are allowed to move this frame
        float allowedStep = _currentSpeedMps * dt; // meters
        if (allowedStep <= 0f) return;

        float remaining = allowedStep;

        // Start from current point on spline
        Vector3 currentPos = FullSpline.GetPosition(_fullT);

        // Iterate forward along the spline in small param steps,
        // without ever exceeding allowedStep in total world distance.
        const int maxIterations = 8;       // safety bound to avoid heavy work per frame
        const float paramStep   = 0.02f;   // step in spline param space per iteration

        for (int i = 0; i < maxIterations && remaining > 0f && _fullT < 1f; i++)
        {
            float candidateT = Mathf.Min(_fullT + paramStep, 1f);
            Vector3 candidatePos = FullSpline.GetPosition(candidateT);
            float segDist = Vector3.Distance(currentPos, candidatePos);

            if (segDist < 0.0001f)
            {
                // Degenerate segment, just advance param
                _fullT = candidateT;
                continue;
            }

            if (segDist > remaining)
            {
                // We cannot go all the way to candidateT this frame without exceeding allowedStep.
                // Go only part of the way, along the chord between currentPos and candidatePos.
                float factor = remaining / segDist;
                Vector3 finalPos = Vector3.Lerp(currentPos, candidatePos, factor);
                float finalT = Mathf.Lerp(_fullT, candidateT, factor);

                myObject.position = finalPos;
                OrientAlongFullSpline(finalT);

                _fullT = finalT;
                remaining = 0f; // we've used up allowedStep
            }
            else
            {
                // We can safely move to candidateT this iteration.
                remaining -= segDist;
                currentPos = candidatePos;
                _fullT = candidateT;

                myObject.position = currentPos;
                OrientAlongFullSpline(_fullT);
            }
        }

        // If we've reached the end of the spline, stop manual movement.
        if (_fullT >= 1f)
        {
            _fullManualActive = false;
            _currentSpeedMps = 0f;
        }
    }

    // ========================= Constant-deceleration yielding (future experiments) =========================

    /// <summary>
    /// Sets up constant-deceleration yielding: builds arc-length tables for the stop and
    /// drive-off splines, computes the braking onset, logs the planned schedule and places
    /// the car at the start of the stop spline.
    /// </summary>
    void StartConstantDecelYield()
    {
        _yStopSpline = (Yield == 2) ? ThirdSpline : FirstSpline;
        _yGoSpline = (Yield == 2) ? FourthSpline : SecondSpline;

        if (_yStopSpline == null || _yGoSpline == null || myObject == null)
        {
            Debug.LogError("CarMovement: stop/drive-off spline or myObject not assigned, cannot run constant-deceleration yielding.");
            _yPhase = YieldPhase.Idle;
            return;
        }

        _yStopLut = BuildArcLengthTable(_yStopSpline, out _yStopLength);
        _yGoLut = BuildArcLengthTable(_yGoSpline, out _yGoLength);

        float vCruise = yieldCruiseSpeedKmh / 3.6f;
        float aAccel = accelKmhPerSec / 3.6f;
        float aBrake = yieldDecelerationMps2;
        float brakeDist = vCruise * vCruise / (2f * aBrake);
        float accelDist = vCruise * vCruise / (2f * aAccel);

        _yBrakeOnsetS = _yStopLength - brakeDist;
        _yBrakeDecelMps2 = aBrake;
        _yS = 0f;
        _ySpeedMps = 0f;
        _yStandstillElapsed = 0f;
        _yPhase = YieldPhase.Approach;
        yielding = false;

        if (_yBrakeOnsetS < accelDist)
        {
            Debug.LogWarning($"CarMovement: stop spline ({_yStopLength:F2} m) is too short to reach {yieldCruiseSpeedKmh:F0} km/h " +
                             $"and brake at {aBrake:F2} m/s^2 (needs {accelDist + brakeDist:F2} m). Lengthen the spline or start later.");
        }

        // Planned schedule (s after trial onset), written to the log for checking against the events
        float tAccel = vCruise / aAccel;
        float tOnset = tAccel + Mathf.Max(0f, _yBrakeOnsetS - accelDist) / vCruise;
        float tStop = tOnset + vCruise / aBrake;
        float tResume = tStop + yieldStandstillS;
        LogLine($"[YIELD_PLAN] constant deceleration {aBrake:F2} m/s^2 from {yieldCruiseSpeedKmh:F2} km/h | " +
                $"braking distance={brakeDist:F2}m | stop spline={_yStopLength:F2}m | drive-off spline={_yGoLength:F2}m | " +
                $"onset t={tOnset:F3}s | stop t={tStop:F3}s | resume t={tResume:F3}s");

        if (tResume > 19f)
            Debug.LogWarning($"CarMovement: planned drive-off at {tResume:F2} s is later than the 19 s yielding trial length in Wave().");

        PlaceOnSpline(_yStopSpline, 0f);
    }

    /// <summary>
    /// Advances constant-deceleration yielding by one physics step. Each phase change is
    /// found exactly within the step, so braking starts, standstill and drive-off happen at
    /// the planned distance and time, and events are logged with exact times.
    /// </summary>
    void UpdateConstantDecelYield()
    {
        if (_yPhase == YieldPhase.Idle || _yPhase == YieldPhase.Done) return;

        float dt = Time.fixedDeltaTime;
        float tStepStart = Time.time - startTime - dt; // trial time at the start of this step
        float rem = dt;                                // time left to integrate in this step
        float vCruise = yieldCruiseSpeedKmh / 3.6f;
        float aAccel = accelKmhPerSec / 3.6f;
        float sBefore = _yS;
        bool onGoSplineBefore = (_yPhase == YieldPhase.DriveOff);

        // Bounded loop: at most one transition per phase within a step
        for (int guard = 0; guard < 8 && rem > 0f; guard++)
        {
            switch (_yPhase)
            {
                case YieldPhase.Approach:
                {
                    float toOnset = _yBrakeOnsetS - _yS;
                    if (toOnset <= 0f)
                    {
                        BeginBraking(tStepStart + (dt - rem));
                        break;
                    }
                    if (_ySpeedMps < vCruise)
                    {
                        // Accelerating toward cruise speed
                        float tToCruise = (vCruise - _ySpeedMps) / aAccel;
                        float tau = Mathf.Min(rem, tToCruise);
                        float d = _ySpeedMps * tau + 0.5f * aAccel * tau * tau;
                        if (d >= toOnset)
                        {
                            float tHit = (-_ySpeedMps + Mathf.Sqrt(_ySpeedMps * _ySpeedMps + 2f * aAccel * toOnset)) / aAccel;
                            _ySpeedMps += aAccel * tHit;
                            _yS = _yBrakeOnsetS;
                            rem -= tHit;
                            BeginBraking(tStepStart + (dt - rem));
                        }
                        else
                        {
                            _ySpeedMps = (tau >= tToCruise) ? vCruise : _ySpeedMps + aAccel * tau;
                            _yS += d;
                            rem -= tau;
                        }
                    }
                    else
                    {
                        // Cruising at constant speed
                        float tHit = toOnset / _ySpeedMps;
                        if (tHit <= rem)
                        {
                            _yS = _yBrakeOnsetS;
                            rem -= tHit;
                            BeginBraking(tStepStart + (dt - rem));
                        }
                        else
                        {
                            _yS += _ySpeedMps * rem;
                            rem = 0f;
                        }
                    }
                    break;
                }

                case YieldPhase.Braking:
                {
                    float tToStop = _ySpeedMps / _yBrakeDecelMps2;
                    if (tToStop <= rem)
                    {
                        _ySpeedMps = 0f;
                        _yS = _yStopLength;
                        rem -= tToStop;
                        _yStandstillElapsed = 0f;
                        _yPhase = YieldPhase.Standstill;
                        yielding = false;
                        LogYieldEvent("YIELD_STOP", tStepStart + (dt - rem));
                        LogYieldEvent("YIELD_END", tStepStart + (dt - rem));
                    }
                    else
                    {
                        _ySpeedMps -= _yBrakeDecelMps2 * rem;
                        // Position from the remaining braking distance, so the car stops exactly at the end
                        _yS = _yStopLength - _ySpeedMps * _ySpeedMps / (2f * _yBrakeDecelMps2);
                        rem = 0f;
                    }
                    break;
                }

                case YieldPhase.Standstill:
                {
                    float need = yieldStandstillS - _yStandstillElapsed;
                    if (need <= rem)
                    {
                        rem -= need;
                        _yStandstillElapsed = yieldStandstillS;
                        _yPhase = YieldPhase.DriveOff;
                        _yS = 0f;
                        LogYieldEvent("YIELD_RESUME", tStepStart + (dt - rem));
                    }
                    else
                    {
                        _yStandstillElapsed += rem;
                        rem = 0f;
                    }
                    break;
                }

                case YieldPhase.DriveOff:
                {
                    float tToCruise = Mathf.Max(0f, (vCruise - _ySpeedMps) / yieldDriveOffAccelMps2);
                    float tau = Mathf.Min(rem, tToCruise);
                    _yS += _ySpeedMps * tau + 0.5f * yieldDriveOffAccelMps2 * tau * tau;
                    _ySpeedMps = (tau >= tToCruise) ? vCruise : _ySpeedMps + yieldDriveOffAccelMps2 * tau;
                    _yS += _ySpeedMps * (rem - tau);
                    rem = 0f;

                    if (_yS >= _yGoLength)
                    {
                        _yS = _yGoLength;
                        _yPhase = YieldPhase.Done;
                    }
                    break;
                }

                default:
                    rem = 0f;
                    break;
            }
        }

        // Place the car and spin the wheels by the distance travelled in this step
        bool onGoSpline = (_yPhase == YieldPhase.DriveOff || _yPhase == YieldPhase.Done);
        if (onGoSpline)
            PlaceOnSpline(_yGoSpline, ArcLengthToT(_yGoLut, _yS));
        else
            PlaceOnSpline(_yStopSpline, ArcLengthToT(_yStopLut, _yS));

        float travelled = (onGoSpline == onGoSplineBefore)
            ? _yS - sBefore
            : (_yStopLength - sBefore) + _yS; // switched splines in this step
        SpinWheels(travelled);
    }

    /// <summary>
    /// Switches to the braking phase. At cruise speed the deceleration is exactly
    /// yieldDecelerationMps2; if the car has not reached cruise speed (spline too short),
    /// the deceleration is adjusted so that it still stops at the end of the stop spline.
    /// </summary>
    void BeginBraking(float tEvent)
    {
        float remaining = _yStopLength - _yS;
        _yBrakeDecelMps2 = yieldDecelerationMps2;
        if (Mathf.Abs(_ySpeedMps - yieldCruiseSpeedKmh / 3.6f) > 0.01f && remaining > 0.01f)
        {
            _yBrakeDecelMps2 = _ySpeedMps * _ySpeedMps / (2f * remaining);
            Debug.LogWarning($"CarMovement: braking started at {_ySpeedMps * 3.6f:F2} km/h, deceleration set to {_yBrakeDecelMps2:F2} m/s^2.");
        }
        _yPhase = YieldPhase.Braking;
        yielding = true; // starts the yielding eHMI animation, as in the original setup
        LogYieldEvent("YIELD_START", tEvent);
    }

    /// <summary>Logs a yield event with its exact trial time and the car state.</summary>
    void LogYieldEvent(string tag, float tEvent)
    {
        string wall = System.DateTime.UtcNow.AddHours(2f).ToString("HH:mm:ss");
        Vector3 p1Pos = conditionScript != null ? conditionScript.p1_object.transform.position : Vector3.zero;
        Vector3 p2Pos = conditionScript != null ? conditionScript.p2_object.transform.position : Vector3.zero;
        LogLine($"[{tag}] t={tEvent:F3}s (wall {wall}) | speed={_ySpeedMps * 3.6f:F2} km/h | " +
                $"decel={_yBrakeDecelMps2:F2} m/s^2 | splineS={_yS:F2}m",
                p1Pos, p2Pos);
    }

    /// <summary>
    /// Trial time at which the car passed a pedestrian, by linear interpolation of the signed
    /// X difference between the previous and the current physics step.
    /// </summary>
    float ExactCrossTime(float tNow, float lastRel, float rel)
    {
        float span = Mathf.Abs(lastRel) + Mathf.Abs(rel);
        if (span < 1e-6f) return tNow;
        return tNow - Time.fixedDeltaTime * Mathf.Abs(rel) / span;
    }

    /// <summary>Cumulative arc length (m) of a spline at t = i / ArcLutSteps.</summary>
    float[] BuildArcLengthTable(Spline spline, out float length)
    {
        float[] lut = new float[ArcLutSteps + 1];
        Vector3 prev = spline.GetPosition(0f);
        lut[0] = 0f;
        for (int i = 1; i <= ArcLutSteps; i++)
        {
            Vector3 p = spline.GetPosition((float)i / ArcLutSteps);
            lut[i] = lut[i - 1] + Vector3.Distance(prev, p);
            prev = p;
        }
        length = lut[ArcLutSteps];
        return lut;
    }

    /// <summary>Spline parameter t for a given arc length s, using the arc-length table.</summary>
    float ArcLengthToT(float[] lut, float s)
    {
        if (s <= 0f) return 0f;
        if (s >= lut[ArcLutSteps]) return 1f;
        int lo = 0, hi = ArcLutSteps;
        while (hi - lo > 1)
        {
            int mid = (lo + hi) / 2;
            if (lut[mid] < s) lo = mid; else hi = mid;
        }
        float seg = lut[hi] - lut[lo];
        float f = seg > 1e-6f ? (s - lut[lo]) / seg : 0f;
        return (lo + f) / ArcLutSteps;
    }

    /// <summary>Places the car at t on a spline, facing along the spline.</summary>
    void PlaceOnSpline(Spline spline, float t)
    {
        Vector3 pos = spline.GetPosition(t);
        myObject.position = pos;

        // Look slightly ahead (or behind at the end) to get the driving direction
        Vector3 dir = (t < 0.999f)
            ? spline.GetPosition(Mathf.Min(t + 0.001f, 1f)) - pos
            : pos - spline.GetPosition(t - 0.001f);
        if (dir.sqrMagnitude > 1e-8f)
            myObject.rotation = Quaternion.LookRotation(dir.normalized, Vector3.up);
    }

    /// <summary>Rotates the wheels to match the distance travelled (m).</summary>
    void SpinWheels(float distance)
    {
        if (distance <= 0f) return;
        float deg = distance / (Mathf.PI * wheelSize) * 360f;
        foreach (GameObject w in new[] { Lfront, Lrear, Rfront, Rrear })
            if (w != null) w.transform.Rotate(Vector3.right, -deg, Space.Self);
    }

    /// <summary>
    /// Orients the car so its forward vector follows the spline tangent at t.
    /// </summary>
    void OrientAlongFullSpline(float t)
    {
        if (FullSpline == null || myObject == null) return;

        float aheadT = Mathf.Min(t + 0.01f, 1f);
        Vector3 pos      = FullSpline.GetPosition(t);
        Vector3 posAhead = FullSpline.GetPosition(aheadT);

        Vector3 dir = (posAhead - pos).normalized;
        if (dir.sqrMagnitude > 0.0001f)
        {
            myObject.position = pos; // ensure we're exactly on the spline
            myObject.rotation = Quaternion.LookRotation(dir, Vector3.up);
        }
    }

    /// <summary>
    /// Utility console output of current car distance and elapsed time (not used by core logic).
    /// </summary>
    void OutputTime()
    {
        Debug.Log(carDistance + "m; " + fixedDeltaTime + "s");
    }

    /// <summary>
    /// Helper class to rotate wheel meshes based on linear travel distance and an easing curve.
    /// </summary>
    public class WheelSpin0
    {
        /// <summary>The wheel GameObject to rotate.</summary>
        public GameObject Wheel;
        /// <summary>Easing curve to apply to rotation tween.</summary>
        public AnimationCurve myAnimation;
        /// <summary>Linear distance represented by this rotation segment (meters).</summary>
        public float Distance;
        /// <summary>Wheel diameter (meters) used to convert distance to degrees.</summary>
        public float WheelDiameter;

        /// <summary>
        /// Constructs a wheel rotation helper.
        /// </summary>
        public WheelSpin0(GameObject trans, AnimationCurve curve, float dist, float wheel)
        {
            Wheel = trans; myAnimation = curve; Distance = dist; WheelDiameter = wheel;
        }

        /// <summary>
        /// Schedules a LeanTween rotateAroundLocal to spin the wheel the appropriate amount.
        /// </summary>
        public void SetupTween(float duration, float delay)
        {
            // Convert linear distance to rotation degrees around local X
            float WheelDist = Mathf.PI * WheelDiameter; // circumference
            float Rotations = Distance / WheelDist;
            float deg = Rotations * 360;

            LeanTween.rotateAroundLocal(Wheel, Vector3.right, -deg, duration)
                .setEase(myAnimation)
                .setDelay(delay);
        }
    }
}
