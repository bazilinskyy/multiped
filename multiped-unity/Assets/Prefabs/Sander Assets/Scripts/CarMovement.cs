using System.Collections;
using System.Collections.Generic;
using System.IO; // For writing .txt logs
using UnityEngine;
using Pixelplacement;
using Pixelplacement.TweenSystem;
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

    /// <summary>Animation curves for the motion timing of different route parts.</summary>
    public AnimationCurve FirstCurve; public AnimationCurve SecondCurve; public AnimationCurve FullCurve;

    // Tween handles for route segments
    TweenBase FirstTween; TweenBase SecondTween; TweenBase ThirdTween; TweenBase FourthTween; TweenBase FullTween;

    // Timings/distances used when yielding to a pedestrian (scenario specific)
    private float firstAni = 11f;   // Duration of first segment when yielding to P1
    private float secDel = 14f;     // Delay before second segment when yielding to P1
    private int firstDist = 115;    // Wheel rotation distance for segment 1 (yield P1)

    /// <summary>Reference point on car route used for distance checks.</summary>
    public GameObject measuringPoint;
    /// <summary>Current distance of car along measurement axis (meters).</summary>
    public float carDistance;

    // Timings for other scenarios and wheel rotation parameters
    private float secAni = 5f;
    private int Ani = 12;
    private float wheelSize = 0.5f;
    private int secDist = 30;
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

    /// <summary>Estimated car speed (km/h), updated by SpeedCalculator.</summary>
    public float speed;

    int counter;                      // Internal counter to gate yield logging events
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

    // ========================= Yield event logging =========================

    [Header("Yield event logging")]
    [Tooltip("Consider the car 'stopped' if speed (km/h) is <= this threshold.")]
    public float stoppedSpeedKmhThreshold = 0.5f;

    bool lastYieldingState = false; // For rising/falling edge detection of yielding
    bool isStopped = false;         // Tracks 'stopped' sub-state within yielding

    // ========================= Speed sampling =========================

    [Header("Speed sampling")]
    [Tooltip("Interval in seconds between speed samples.")]
    public float speedSampleInterval = 0.5f;

    private Vector3 _lastSpeedSamplePos;
    private float _lastSpeedSampleTime;
    private bool _speedInitialized = false;

    // ========================= Speed limiting (non-yield) =========================

    [Header("Speed limit (non-yield)")]
    [Tooltip("Maximum physical speed the car is allowed to reach in km/h when not yielding.")]
    public float maxSpeedKmh = 50f;

    [Tooltip("Acceleration toward max speed in km/h per second (non-yield).")]
    public float accelKmhPerSec = 40f;

    // Internal state for manual movement on FullSpline (non-yield)
    bool _fullManualActive = false;
    float _fullT = 0f;               // current param on [0,1] along FullSpline
    float _currentSpeedMps = 0f;     // current speed in m/s (for movement, not logging)

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
    /// Unity Start: initialize speed sampling (logging already initialized in Awake).
    /// </summary>
    void Start()
    {
        if (distance_cube != null)
        {
            _lastSpeedSamplePos = distance_cube.transform.position;
            _lastSpeedSampleTime = Time.time;
            _speedInitialized = true;
        }
        else
        {
            _speedInitialized = false;
            Debug.LogWarning("CarMovement: distance_cube is not assigned. Speed sampling will be disabled until it is set.");
        }
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

        // Update speed estimate (sampled every speedSampleInterval seconds)
        SpeedCalculator();

        // Continuous per-tick console log (verbose)
        Debug.Log($"[car_t={(Time.time - startTime):F2}s] continous_pedestrian1_distance= {pedestrian1_distance_x} pedestrian2_distance= {pedestrian2_distance_x}");

        // --------------------------- Legacy yielding gates (counter-based) ---------------------------
        if (pedestrian2_distance < 43 && Yield == 1)
        {
            counter += 1;
            if (counter == 1)
            {
                yielding = true;
                Debug.Log("Start yielding at: " + fixedDeltaTime + "; Distance: " + carDistance);
                Debug.Log("distance cube: " + distance_cube.transform.position + "P1_distance: " + conditionScript.p1_object.transform.position + "P2_distance: " + conditionScript.p2_object.transform.position);
                Debug.Log("pedestrian1_distance= " + pedestrian1_distance + " pedestrian2_distance=" + pedestrian2_distance);
            }

            if (pedestrian2_distance < 3)
            {
                yielding = false;
                Debug.Log("2nd_distance cube: " + distance_cube.transform.position + "P1_distance: " + conditionScript.p1_object.transform.position + "P2_distance: " + conditionScript.p2_object.transform.position);
                Debug.Log("pedestrian1_distance= " + pedestrian1_distance + " pedestrian2_distance=" + pedestrian2_distance);
            }
        }
        if (pedestrian1_distance < 43 && Yield == 2)
        {
            counter += 1;
            if (counter == 1)
            {
                yielding = true;
            }

            if (pedestrian1_distance < 3)
            {
                yielding = false;
            }
        }
        // -------------------------------------------------------------------------------------------

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
                    LogLine($"[CROSS] P1 at t={t:F2}s (wall {wallClock}) | carDist={carDistance:F1}m | speed={speed:F1} km/h | pedDistance={pedestrianPairDistance:F2}m",
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
                    LogLine($"[CROSS] P2 at t={t:F2}s (wall {wallClock}) | carDist={carDistance:F1}m | speed={speed:F1} km/h | pedDistance={pedestrianPairDistance:F2}m",
                            p1Pos, p2Pos);
                }
            }
            lastRelX2 = relX2;
        }
        // ========================================================================================================

        // ====================== Yield event logging (start/stop/resume/end) ======================
        float tCar = Time.time - startTime;
        string wall = System.DateTime.UtcNow.AddHours(2f).ToString("HH:mm:ss");

        // Yield just started this frame
        if (yielding && !lastYieldingState)
        {
            LogLine($"[YIELD_START] t={tCar:F2}s (wall {wall}) | " +
                    $"carDist={carDistance:F2}m | speed={speed:F2} km/h | " +
                    $"dP1={pedestrian1_distance:F2}m dP2={pedestrian2_distance:F2}m",
                    p1Pos, p2Pos);
            isStopped = false;
        }

        // While yielding, detect when we actually come to a stop and when we resume
        if (yielding)
        {
            if (!isStopped && speed <= stoppedSpeedKmhThreshold)
            {
                isStopped = true;
                LogLine($"[YIELD_STOP]  t={tCar:F2}s (wall {wall}) | " +
                        $"carDist={carDistance:F2}m | speed={speed:F2} km/h | " +
                        $"dP1={pedestrian1_distance:F2}m dP2={pedestrian2_distance:F2}m",
                        p1Pos, p2Pos);
            }
            else if (isStopped && speed > stoppedSpeedKmhThreshold)
            {
                isStopped = false;
                LogLine($"[YIELD_RESUME] t={tCar:F2}s (wall {wall}) | " +
                        $"carDist={carDistance:F2}m | speed={speed:F2} km/h | " +
                        $"dP1={pedestrian1_distance:F2}m dP2={pedestrian2_distance:F2}m",
                        p1Pos, p2Pos);
            }
        }

        // Yield just ended this frame
        if (!yielding && lastYieldingState)
        {
            LogLine($"[YIELD_END]   t={tCar:F2}s (wall {wall}) | " +
                    $"carDist={carDistance:F2}m | speed={speed:F2} km/h | " +
                    $"dP1={pedestrian1_distance:F2}m dP2={pedestrian2_distance:F2}m",
                    p1Pos, p2Pos);
            isStopped = false;
        }

        // Keep last state for edge detection next tick
        lastYieldingState = yielding;
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

            // Reset yield gate counter for next car
            counter = 0;
        }
    }

    /// <summary>
    /// Starts spline tweens and wheel rotations for the current Yield mode.
    /// Non-yield case uses manual movement with a physical speed cap.
    /// </summary>
    public void DriveCar()
    {
        if (Yield > 0)
        {
            if (Yield == 1)
            {
                // Parameters for yielding to pedestrian 1
                firstAni = 11f;
                secDel = 14f;
                firstDist = 115;
                secAni = 5f;
                secDist = 30;

                // Two-segment route
                FirstTween = Tween.Spline(FirstSpline, myObject, 0, 1, true, firstAni, 0, FirstCurve, Tween.LoopType.None);
                SecondTween = Tween.Spline(SecondSpline, myObject, 0, 1, true, secAni, secDel, SecondCurve, Tween.LoopType.None);

                // Wheel rotations for both segments
                WheelSpin0 LF = new WheelSpin0(Lfront, FirstCurve, firstDist, wheelSize); LF.SetupTween(firstAni, 0);
                WheelSpin0 LR = new WheelSpin0(Lrear, FirstCurve, firstDist, wheelSize); LR.SetupTween(firstAni, 0);
                WheelSpin0 RF = new WheelSpin0(Rfront, FirstCurve, firstDist, wheelSize); RF.SetupTween(firstAni, 0);
                WheelSpin0 RR = new WheelSpin0(Rrear, FirstCurve, firstDist, wheelSize); RR.SetupTween(firstAni, 0);
                WheelSpin0 LF2 = new WheelSpin0(Lfront, FirstCurve, secDist, wheelSize); LF2.SetupTween(secAni, secDel);
                WheelSpin0 LR2 = new WheelSpin0(Lrear, FirstCurve, secDist, wheelSize); LR2.SetupTween(secAni, secDel);
                WheelSpin0 RF2 = new WheelSpin0(Rfront, FirstCurve, secDist, wheelSize); RF2.SetupTween(secAni, secDel);
                WheelSpin0 RR2 = new WheelSpin0(Rrear, FirstCurve, secDist, wheelSize); RR2.SetupTween(secAni, secDel);
            }
            if (Yield == 2)
            {
                // Parameters for yielding to pedestrian 2
                firstAni = 12f;
                secDel = 15f;
                firstDist = 125;

                secAni = 4f;
                secDist = 20;

                // Two-segment route
                ThirdTween = Tween.Spline(ThirdSpline, myObject, 0, 1, true, firstAni, 0, FirstCurve, Tween.LoopType.None);
                FourthTween = Tween.Spline(FourthSpline, myObject, 0, 1, true, secAni, secDel, SecondCurve, Tween.LoopType.None);

                // Wheel rotations for both segments
                WheelSpin0 LF = new WheelSpin0(Lfront, FirstCurve, firstDist, wheelSize); LF.SetupTween(firstAni, 0);
                WheelSpin0 LR = new WheelSpin0(Lrear, FirstCurve, firstDist, wheelSize); LR.SetupTween(firstAni, 0);
                WheelSpin0 RF = new WheelSpin0(Rfront, FirstCurve, firstDist, wheelSize); RF.SetupTween(firstAni, 0);
                WheelSpin0 RR = new WheelSpin0(Rrear, FirstCurve, firstDist, wheelSize); RR.SetupTween(firstAni, 0);
                WheelSpin0 LF2 = new WheelSpin0(Lfront, FirstCurve, secDist, wheelSize); LF2.SetupTween(secAni, secDel);
                WheelSpin0 LR2 = new WheelSpin0(Lrear, FirstCurve, secDist, wheelSize); LR2.SetupTween(secAni, secDel);
                WheelSpin0 RF2 = new WheelSpin0(Rfront, FirstCurve, secDist, wheelSize); RF2.SetupTween(secAni, secDel);
                WheelSpin0 RR2 = new WheelSpin0(Rrear, FirstCurve, secDist, wheelSize); RR2.SetupTween(secAni, secDel);
            }
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
    /// Estimates vehicle speed by sampling the car proxy position at a fixed interval.
    /// Runs every FixedUpdate without coroutines.
    /// </summary>
    void SpeedCalculator()
    {
        if (!_speedInitialized || distance_cube == null)
            return;

        // If no car is currently running, keep speed at 0 and reset the sample
        if (!WaveStarted)
        {
            speed = 0f;
            _lastSpeedSamplePos = distance_cube.transform.position;
            _lastSpeedSampleTime = Time.time;
            return;
        }

        float now = Time.time;
        float dt = now - _lastSpeedSampleTime;

        // Only update when enough time has passed
        if (dt < speedSampleInterval)
            return;

        Vector3 currentPos = distance_cube.transform.position;
        float distance = Vector3.Distance(_lastSpeedSamplePos, currentPos); // meters

        // m/s = distance / dt, km/h = m/s * 3.6
        float metersPerSecond = distance / dt;
        speed = metersPerSecond * 3.6f;

        _lastSpeedSamplePos = currentPos;
        _lastSpeedSampleTime = now;
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
