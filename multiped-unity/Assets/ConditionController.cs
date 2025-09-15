using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using System;
using UnityEngine.XR;
using System.IO;
using UnityEditor.Recorder;           // NOTE: Editor-only dependency; safe in Editor, remove for builds if unused
using UnityEditor.Recorder.Input;     // NOTE: Editor-only dependency; safe in Editor, remove for builds if unused
using System.Linq;

/// <summary>
/// Serializable data model for a single trial (i.e., one condition run).
/// Rows are typically loaded from mapping.csv via CSVSerializer.
/// </summary>
[System.Serializable]
public class Trial {
    public int no;             // Trial sequence number (index)
    public string scenario;    // Scenario descriptor
    public string video_id;    // Unique identifier for the recorded video (ALSO used for per-trial CSV file name)
    public int yielding;       // 0: yield P1, 1: yield P2, 2: no yield (NOTE: differs from CarMovement's 0/1/2 semantics; ensure mapping aligns)
    public int eHMIOn;         // 0: off, 1: on (light band)
    public int distPed;        // Pedestrian separation control (domain-specific; used to compute X offsets)
    public int p1;             // Presence of Pedestrian 1 (0 off / 1 on)
    public int p2;             // Presence of Pedestrian 2 (0 off / 1 on)
    public int camera;         // Camera placement preset selector
    public int group;          // Optional subject group flag (not used here)
    public int video_length;   // Target duration (ms) for UI_duration wait → then questions
}

/// <summary>
/// Orchestrates each trial/condition:
/// - Loads/shuffles mapping.csv describing trials
/// - Positions pedestrians and camera based on the active trial
/// - Starts the car behavior (via CarMovement)
/// - Runs timed UI flow (black screen, countdowns, questions)
/// - Records per-frame XR inputs to CSV (one file per trial, named by video_id)
/// - Writes questionnaire answers to a separate CSV
/// - Advances through all trials and exits
/// </summary>
public class ConditionController : MonoBehaviour
{
    // =========================
    // File naming & paths
    // =========================

    /// <summary>
    /// Prefix for questionnaire output CSV files (e.g., "a_mapping.csv", "a_YYYYMMDD_HHMMSS.csv").
    /// </summary>
    public string writeFileName = "a"; // "a" for standard; change per subject/session if needed

    /// <summary>
    /// Full path for the main questionnaire CSV written during the session.
    /// </summary>
    public string writeFilePath = "";

    // =========================
    // Experiment flow fields
    // =========================

    private float initialHeight;         // Cache of camera Y to prevent vertical drift in VR
    public bool conditionFinished = false;

    private LightStripBumper lightStripBumper; // eHMI lightstrip script (if used)
    public GameObject LEDBumperObject;         // Light strip root object (toggled on at start)
    public GameObject tracker;                 // (unused in this file) external tracker object
    public GameObject progress;                // (unused in this file) progress bar object
    public GameObject projection;              // (unused in this file) projection object

    private CarMovement carMovementScript;     // Reference to car controller
    private PlayFabController playfabScript;   // Reference to PlayFab wrapper
    public int conditionCounter = 0;           // Current trial index
    public int numberConditions = 0;           // Total number of trials loaded

    // Trial parameters (populated from trials[conditionCounter])
    public int eHMIOn = 0;     // 0=no, 1=slowly-pulsing light band
    public int yielding = 0;   // Mapping definition: 0 yes for P1 / 1 yes for P2 / 2 no
    public int distPed = 0;    // Distance parameter that shifts peds along X
    public int p1 = 0;         // Presence P1 (0/1)
    public int p2 = 0;         // Presence P2 (0/1)
    public int camera = 0;     // Camera preset (0..4)
    public int duration = 0;   // Trial duration in milliseconds for UI wait before questions

    // UI references for demo flow
    public GameObject demoWelcomeCanvas;
    public GameObject demoWalkCanvas;
    public GameObject demoInfoCanvas1;
    public GameObject demoInfoCanvas2;

    // UI references for trial flow
    public GameObject trialWalkCanvas;
    public GameObject trialDemoCanvas;
    public GameObject trialStartCanvas;
    public GameObject trialEndCanvas;
    public GameObject ExperimentEndCanvas;

    public Text demoTitle;
    public Text demoText;
    public Text trialTitle;

    // Misc. UI and state
    public GameObject WillingnessToCross;  // Overlay UI during car run (reticle + question)
    public GameObject reticle;             // Gaze reticle
    public GameObject CountDown;           // 3..2..1.. UI
    public bool preview = false;           // Preview state flag
    public bool trial = false;             // Trial state flag

    public AudioSource buttonSound;        // Generic SFX

    /// <summary>
    /// All trials loaded from mapping.csv.
    /// </summary>
    public Trial[] trials;

    // Scene references (set in inspector)
    public GameObject p1_object;      // Pedestrian 1 GameObject
    public GameObject p2_object;      // Pedestrian 2 GameObject
    public GameObject camera_object;  // Camera rig transform to reposition
    public GameObject black_canvas;   // Fullscreen black UI canvas for interstitial

    private int trialsPassed = 0;     // Not used here; reserved for bookkeeping

    // =========================
    // Unity Lifecycle
    // =========================

    /// <summary>
    /// Bootstraps the session:
    /// - Shuffles mapping.csv (except first three lines)
    /// - Copies the shuffled file into Assets for archival
    /// - Sets unique path for per-session questionnaire CSV (writeFilePath)
    /// - Loads trials via CSVSerializer
    /// - Starts VR activation coroutine (currently no-ops)
    /// - Proceeds to Start2() to initialize the first trial
    /// </summary>
    public void Start()
    {
        // The original mapping.csv is stored two directories above Assets
        string filePath = Application.dataPath + "/../../mapping.csv";

        // Shuffle rows after the header block; writes back to mapping.csv in-place
        ShuffleCSVFile();

        // Make a copy of mapping.csv next to Assets for reproducibility
        string mappingFilePath = Application.dataPath + "/" + writeFileName + "_mapping.csv";
        try
        {
            File.Copy(filePath, mappingFilePath, true);
            Debug.Log("CSV file copied successfully to: " + mappingFilePath);
        }
        catch (Exception e)
        {
            Debug.LogError("Failed to copy the CSV file: " + e.Message);
        }

        // Create a unique per-session questionnaire CSV path (timestamped)
        writeFilePath = Application.dataPath + "/" +  writeFileName + "_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") +  ".csv";

        // If an old file with same name exists (rare), delete to start fresh
        if (File.Exists(writeFilePath))
        {
            Debug.Log("File already exists, deleting...");
            File.Delete(writeFilePath);
        }

        Debug.Log("Start");

        // Load the trials into memory
        string text = File.ReadAllText(filePath);
        trials = CSVSerializer.Deserialize<Trial>(text);
        numberConditions = trials.Length;
        Debug.Log("Number of conditions: " + numberConditions);

        // Initialize VR (currently disabled to avoid XR device load during tests)
        StartCoroutine(ActivatorVR("cardboard"));

        // Cache SFX
        buttonSound = GetComponent<AudioSource>();

        // Proceed to trial setup
        Start2();
    }

    /// <summary>
    /// Shuffles mapping.csv rows after the first three lines.
    /// Ensures reproducible randomization using System.Random.
    /// </summary>
    public void ShuffleCSVFile()
    {
        string filePath = Application.dataPath + "/../../mapping.csv";

        if (!File.Exists(filePath))
        {
            Debug.LogError("CSV file not found at: " + filePath);
            return;
        }

        // Read entire file into memory
        var lines = File.ReadAllLines(filePath).ToList();
        if (lines.Count <= 3)
        {
            Debug.LogError("The CSV file does not have enough rows to shuffle.");
            return;
        }

        // Preserve the first three rows, shuffle the rest
        var firstThreeRows = lines.Take(3).ToList();
        var data = lines.Skip(3).ToList();

        // Shuffle with System.Random
        System.Random rand = new System.Random();
        var shuffledData = data.OrderBy(x => rand.Next()).ToList();

        // Recombine and overwrite file
        var result = firstThreeRows.Concat(shuffledData).ToList();
        File.WriteAllLines(filePath, result);

        Debug.Log("CSV rows have been shuffled and saved to: " + filePath);
    }

    /// <summary>
    /// Stub: would normally load/enable an XR device by name.
    /// Currently a no-op to avoid XR issues during dev.
    /// </summary>
    public IEnumerator ActivatorVR(string YESVR)
    {
        // XRSettings.LoadDeviceByName(YESVR);
        yield return null;
        // XRSettings.enabled = true;
    }

    /// <summary>
    /// Stub: would normally unload/disable XR device by name.
    /// Currently a no-op to avoid XR issues during dev.
    /// </summary>
    public IEnumerator DectivatorVR(string NOVR)
    {
        // XRSettings.LoadDeviceByName(NOVR);
        yield return null;
        // XRSettings.enabled = false;
    }

    // =========================
    // Trial setup & camera/ped layout
    // =========================

    public float time1, time2 = 0f;    // Session timing (used for univ_timestamp)
    // Default spawn points used by ResetPositions()
    Vector3 initialPositionP1 = new Vector3(105.792694f,-3.31599998f,3.44000006f);
    Vector3 initialPositionP2 = new Vector3(104.082703f,-3.31599998f,3.44000006f);

    /// <summary>
    /// Returns pedestrians to canonical positions at the start of a trial.
    /// </summary>
    void ResetPositions() {
        p1_object.transform.position = initialPositionP1;
        p2_object.transform.position = initialPositionP2;
    }

    public GameObject player1, player2; // not used here, but left for clarity

    /// <summary>
    /// Initializes one trial (based on conditionCounter):
    /// - Resets positions
    /// - Applies trial parameters (presence, spacing, camera preset)
    /// - Shows black screen briefly
    /// - Starts the car and UI timing
    /// - Locks camera vertical drift
    /// </summary>
    void Start2()
    {
        ResetPositions();

        // Zero the base timestamp used by UpdateInputDataDisplay()
        time1 = Time.time;

        // Cache scene scripts
        carMovementScript = GameObject.Find("CarMovement").GetComponent<CarMovement>();
        playfabScript     = GameObject.Find("PlayFabController").GetComponent<PlayFabController>();

        // UI/FX defaults per trial
        LEDBumperObject.SetActive(true);
        tracker.SetActive(false);
        progress.SetActive(false);
        projection.SetActive(false);

        // Pull the active trial parameters
        eHMIOn  = trials[conditionCounter].eHMIOn;
        yielding= trials[conditionCounter].yielding;
        distPed = trials[conditionCounter].distPed;
        p1      = trials[conditionCounter].p1;
        p2      = trials[conditionCounter].p2;
        camera  = trials[conditionCounter].camera;
        duration= trials[conditionCounter].video_length;

        Debug.Log(conditionCounter +  ":: eHMIOn=" + eHMIOn +  " yielding=" + yielding +  " distPed=" + distPed +
          " p1=" + p1 +  " p2=" + p2 + " camera=" + camera);

        // Toggle P1 presence
        if (p1 == 0) {
            p1_object.SetActive(false);
            Debug.Log("P1 disabled");
        } else {
            p1_object.SetActive(true);
            Debug.Log("P1 enabled");
        }

        // Toggle P2 presence
        if (p2 == 0) {
            p2_object.SetActive(false);
            Debug.Log("P2 disabled");
        } else {
            p2_object.SetActive(true);
            Debug.Log("P2 enabled");
        }

        // --- Position pedestrians according to distPed ---
        // We shift P2 left of P1 by 2*distPed, then apply a shared adjustment to both
        Debug.Log("Case start - P1: " + p1_object.transform.position + ", P2: " + p2_object.transform.position);
        float deltaDist   = 2f * distPed;          // base spacing along X
        float adjustment  = (5.0f - distPed) * 2;  // global offset applied to both (scene-specific tuning)
        if (distPed != 0) {
            // Place P2 distPed units (scaled) left of P1 in X
            p2_object.transform.position = new Vector3(p1_object.transform.position.x - deltaDist,
                                                       p1_object.transform.position.y,
                                                       p1_object.transform.position.z);

            Debug.Log("adjustment" + adjustment);

            // Global offset to pull both backward consistently with camera/scene layout
            p1_object.transform.position = new Vector3(p1_object.transform.position.x - adjustment,
                                                       p1_object.transform.position.y,
                                                       p1_object.transform.position.z);

            p2_object.transform.position = new Vector3(p2_object.transform.position.x - adjustment,
                                                       p2_object.transform.position.y,
                                                       p2_object.transform.position.z);

            Debug.Log("distance between pedestrians set to distPed=" + distPed + ": (posP1.x - " + 2 * distPed +
                ", posP1.y, posP1.z). coordinates of P2=(" + p2_object.transform.position.x + ", " +
                p2_object.transform.position.y + ", " + p2_object.transform.position.z + ")");
        } else {
            Debug.Log("distance between pedestrians not set (distPed=0)");
        }

        // --- Camera placement presets ---
        // P1 head, P2 head, 3rd-person, and two "moving away" variants.
        Vector3 posCameraP1   = new Vector3(105.54f, -1.717f, 3.6f);
        Vector3 rotCameraP1   = new Vector3(0f, 0f, 0f);
        Vector3 rotCameraP2   = new Vector3(0f, 0f, 0f);
        Vector3 posCamera3rd  = new Vector3(108.53f, -0.47f, -2.68f);
        Vector3 rotCamera3rd  = new Vector3(0f, -49.995f, 0f);

        Vector3 targetCameraPos = new Vector3(); // used for interpolated moves
        float transitionDuration = 0.0f;         // seconds to move camera with Lerp

        if (camera == 0) {
            // Pin to P1 head (with global offset to reflect distPed)
            camera_object.transform.position   = posCameraP1;
            camera_object.transform.eulerAngles= rotCameraP1;
            camera_object.transform.position   = new Vector3(posCameraP1.x - ((5.0f - (distPed))*2),
                                                            posCameraP1.y,
                                                            posCameraP1.z);
            Debug.Log("Camera set to head of P1.");
        } else if (camera == 1) {
            // Snap near P2, then push farther left by a fixed 10 in X
            Vector3 intermediateCameraPos = new Vector3(posCameraP1.x - deltaDist, posCameraP1.y, posCameraP1.z);
            camera_object.transform.position    = intermediateCameraPos;
            camera_object.transform.eulerAngles = rotCameraP1;

            Vector3 finalCameraPos = new Vector3(posCameraP1.x - 10.0f, posCameraP1.y, posCameraP1.z);
            camera_object.transform.position = finalCameraPos;
            Debug.Log("Camera set to head of P2.");
        } else if (camera == 2) {
            // 3rd-person vantage
            camera_object.transform.position    = posCamera3rd;
            camera_object.transform.eulerAngles = rotCameraP2;
            Debug.Log("Camera set to 3rd person view.");
        } else if (camera == 3) {
            // Start nearer to P2, then slide towards P1
            camera_object.transform.position    = new Vector3(posCameraP1.x - deltaDist, posCameraP1.y, posCameraP1.z);
            camera_object.transform.eulerAngles = rotCameraP2;
            targetCameraPos     = posCameraP1;
            transitionDuration  = 0.5f * deltaDist; // proportional travel time
            Debug.Log("Camera set to P1 with going away from P2.");
        } else if (camera == 4) {
            // Start at P1, then slide to P2
            camera_object.transform.position    = posCameraP1;
            camera_object.transform.eulerAngles = rotCameraP1;
            targetCameraPos     = new Vector3(posCameraP1.x - deltaDist, posCameraP1.y, posCameraP1.z);
            transitionDuration  = 0.5f * deltaDist;
            Debug.Log("Camera set to P2 with going away from P1.");
        } else {
            Debug.Log("Wrong value for camera given.");
        }

        // Fade to/from black before motion begins (helps avoid pops)
        StartCoroutine(BlackScreen(1f));

        // Kick off the car + optional camera move for this trial
        TrialStart(targetCameraPos, transitionDuration);

        // Gate progression until questionnaire writes (see FixedUpdate)
        startNextStage = false;

        // Launch a coroutine that waits "duration" ms then opens question UI
        StartCoroutine(UI_duration(duration));

        // Prevent camera Y drift in VR (reset in FixedUpdate)
        initialHeight = Camera.main.transform.position.y;
    }

    /// <summary>
    /// Waits for the given duration (ms) then shows Question 1.
    /// </summary>
    IEnumerator UI_duration(int time_duration)
    {
        yield return new WaitForSeconds(time_duration/1000f);
        Question1();
    }

    /// <summary>
    /// Displays a black overlay for t seconds (fade in/out not implemented here).
    /// </summary>
    IEnumerator BlackScreen(float t)
    {
        black_canvas.GetComponent<Image>().color = new Color(0, 0, 0, 255);
        yield return new WaitForSeconds(t);
        black_canvas.GetComponent<Image>().color = new Color(0, 0, 0, 0);
    }

    /// <summary>
    /// Physics tick:
    /// - Pins camera Y to initialHeight to avoid vertical drift
    /// - Checks if current trial's car wave finished → advances to next trial
    /// - Triggers data writing when startNextStage toggles
    /// - Streams XR input to per-frame CSV buffer
    /// </summary>
    private void FixedUpdate()
    {
        // Lock camera vertical axis to avoid user bobbing (VR comfort aid)
        Vector3 currentPosition = Camera.main.transform.position;
        if (Mathf.Abs(currentPosition.y - initialHeight) > 0.01f)
        {
            currentPosition.y = initialHeight;
            Camera.main.transform.position = currentPosition;
        }

        // Flag that tells UpdateInputDataDisplay to flush data to CSV this frame
        if (startNextStage == true)
        {
            write_data = true;
        }

        // Stream XR inputs into csvData buffer each frame
        UpdateInputDataDisplay();

        // If the car reported trial finished & we already collected answers, advance
        if (carMovementScript != null) {
            if (carMovementScript.conditionFinished && startNextStage == true)
            {
                // If this was the last condition, archive files and exit
                if (conditionCounter == numberConditions - 1) {
                    Debug.Log("Experiment finished");
                    SaveAllCSVFiles();
                    SceneManager.LoadScene("EndMenu");
                    Application.Quit();
                }

                Debug.Log("FixedUpdate::trial end");
                WillingnessToCross.SetActive(false);
                reticle.SetActive(true);
                carMovementScript.conditionFinished = false;
                trial = false;

                // Bump to next condition
                conditionCounter = conditionCounter + 1;
                trialEndCanvas.SetActive(false);
                StartCoroutine(ActivatorVR("none"));

                // Start the next trial immediately
                if (startNextStage == true)
                    Start2();
            }
        }
    }

    /// <summary>
    /// Moves all session CSVs produced under Assets/*.csv into data/&lt;writeFileName&gt;/.
    /// Uses File.Move to relocate the files.
    /// </summary>
    private void SaveAllCSVFiles()
    {
        string targetFolderPath = Path.Combine(Application.dataPath, "../../data/" + writeFileName);

        if (!Directory.Exists(targetFolderPath))
        {
            Directory.CreateDirectory(targetFolderPath);
            Debug.Log("Created folder: " + targetFolderPath);
        }

        // Find all CSVs currently under Assets
        string[] csvFiles = Directory.GetFiles(Application.dataPath, "*.csv");

        foreach (string filePath in csvFiles)
        {
            try
            {
                string fileName = Path.GetFileName(filePath);
                string destinationPath = Path.Combine(targetFolderPath, fileName);

                // Move file to target folder
                File.Move(filePath, destinationPath);
                Debug.Log("Copied file: " + fileName + " to " + targetFolderPath);
            }
            catch (Exception e)
            {
                Debug.LogError("Failed to copy file: " + filePath + ". Error: " + e.Message);
            }
        }
    }

    // =========================
    // DEMO UI flow
    // =========================

    void DemoStart()
    {
        Debug.Log("DemoStart");
        demoWelcomeCanvas.SetActive(true);
    }

    public void DemoCanvas1()
    {
        Debug.Log("DemoCanvas1");
        demoWelcomeCanvas.SetActive(false);
        demoWalkCanvas.SetActive(true);
    }

    public void DemoCanvas2()
    {
        Debug.Log("DemoCanvas2");
        demoWalkCanvas.SetActive(false);
        StartCoroutine(WalkForward());
        demoInfoCanvas1.SetActive(true);
    }

    public void DemoCanvas3()
    {
        Debug.Log("DemoCanvas3");
        demoInfoCanvas1.SetActive(false);
        StartCoroutine(CountDownDemo());
    }

    public void DemoCanvas4()
    {
        Debug.Log("DemoCanvas4");
        demoInfoCanvas2.SetActive(false);
        StartCoroutine(ActivatorVR("none"));
        SceneManager.LoadScene("Environment");
    }

    /// <summary>
    /// 3..2..1.. → starts demo car run and shows the Willingness overlay.
    /// </summary>
    IEnumerator CountDownDemo()
    {
        Debug.Log("CountDownDemo");
        reticle.SetActive(false);
        CountDown.SetActive(true);
        carMovementScript.CountSound.Play();
        yield return new WaitForSecondsRealtime(3f);
        carMovementScript.AudioBeep.Play();
        yield return new WaitForSecondsRealtime(1f);
        carMovementScript.StartCarDemo();
        WillingnessToCross.SetActive(true);
    }

    /// <summary>
    /// Small forward walk animation by toggling CameraHolder.MoveCamera.StartWalk.
    /// </summary>
    IEnumerator WalkForward()
    {
        Debug.Log("WalkForward");
        yield return new WaitForSecondsRealtime(0.2f);
        GameObject.Find("CameraHolder").GetComponent<MoveCamera>().StartWalk = true;
        yield return new WaitForSecondsRealtime(3.0f);
    }

    // =========================
    // TRIAL UI flow
    // =========================

    /// <summary>
    /// Entry into trial start flow (camera move + car start).
    /// </summary>
    void TrialStart(Vector3 targetCameraPos, float transitionDuration)
    {
        Debug.Log("TrialStart");
        TrialCanvas3(targetCameraPos, transitionDuration);
    }

    public void TrialCanvas1()
    {
        Debug.Log("TrialCanvas1");
        trialWalkCanvas.SetActive(false);
        StartCoroutine(WalkForward());
        trialDemoCanvas.SetActive(true);
    }

    /// <summary>
    /// Preview flow (no data writing), plays car preview.
    /// </summary>
    public void TrialCanvas2()
    {
        Debug.Log("TrialCanvas2");
        trialDemoCanvas.SetActive(false);
        preview = true;
        StartCoroutine(CountDownPreview());
    }

    IEnumerator CountDownPreview()
    {
        Debug.Log("CountDownPreview");
        reticle.SetActive(false);
        CountDown.SetActive(true);
        carMovementScript.CountSound.Play();
        yield return new WaitForSecondsRealtime(3f);
        carMovementScript.AudioBeep.Play();
        yield return new WaitForSecondsRealtime(1f);
        carMovementScript.StartCarPreview();
        WillingnessToCross.SetActive(true);
    }

    /// <summary>
    /// Starts an actual trial:
    /// - Optional camera interpolation (MoveCamera)
    /// - Set trial=true and start the car route
    /// </summary>
    public void TrialCanvas3(Vector3 targetCameraPos, float transitionDuration)
    {
        Debug.Log("TrialCanvas3");
        carMovementScript.AudioBeep.Play();

        if (transitionDuration > 0) {
            StartCoroutine(MoveCamera(targetCameraPos, transitionDuration));
        }

        trial = true;
        carMovementScript.StartCar();
    }

    /// <summary>
    /// Moves camera from current position to target over transitionDuration using Lerp.
    /// </summary>
    IEnumerator MoveCamera(Vector3 targetCameraPos, float transitionDuration) {
        Debug.Log("Moving camera");
        yield return new WaitForSeconds(1f);
        float t = 0.0f;
        while (t < 1.0f)
        {
            t += Time.deltaTime * (Time.timeScale/transitionDuration);
            camera_object.transform.position = Vector3.Lerp(camera_object.transform.position, targetCameraPos, t);
            yield return null;
        }
    }

    IEnumerator CountDownTrial()
    {
        Debug.Log("CountDownTrial");
        reticle.SetActive(false);
        CountDown.SetActive(true);
        carMovementScript.CountSound.Play();
        yield return new WaitForSecondsRealtime(3f);
        carMovementScript.AudioBeep.Play();
        yield return new WaitForSecondsRealtime(1f);
        playfabScript.deltaTime2 = Time.time;
        trial = true;
        carMovementScript.StartCar();
        WillingnessToCross.SetActive(true);
    }

    public void TrialCanvas4()
    {
        Debug.Log("TrialCanvas4");
        trialEndCanvas.SetActive(false);
        StartCoroutine(ActivatorVR("none"));
        SceneManager.LoadScene("Environment");
    }

    public void RestartPreview()
    {
        Debug.Log("RestartPreview");
        trialStartCanvas.SetActive(false);
        trialDemoCanvas.SetActive(true);
    }

    public void Reset0()
    {
        Debug.Log("Reset0");
        PlayerPrefs.SetInt("Condition Counter", 1);
        StartCoroutine(ActivatorVR("none"));
        SceneManager.LoadScene("Environment");
    }

    public void ButtonSound()
    {
        Debug.Log("ButtonSound");
        buttonSound.Play();
    }

    // =========================
    // Questionnaire CSV writing
    // =========================

    /// <summary>
    /// Writes the questionnaire CSV:
    /// - First creates/overwrites the file with header
    /// - Writes all lines from mainData
    /// </summary>
    public void writeCSV()
    {
        Debug.Log("----------file writing triggered");

        // Overwrite with headers
        TextWriter tw = new StreamWriter(writeFilePath, false);
        tw.WriteLine("Video ID, Answer1, Answer2, Answer3");
        tw.Close();

        // Write all collected answers (mainData rows)
        File.WriteAllLines(writeFilePath, mainData);

        // tw.Close(); // already closed above
    }

    public int answer_element = 0;          // index in trials[] matching answers
    public GameObject Q1, Q2, Q3;           // Question panels
    public GameObject stop1, start_study;   // Pause intermissions
    public Slider slider1, slider2, slider3;// Three answers as sliders 0..1
    public bool startNextStage = false;     // When true → FixedUpdate will trigger next trial

    /// <summary>
    /// Shows Q1 panel and stamps answer_element to current trial index.
    /// </summary>
    public void Question1()
    {
        Q1.SetActive(true);
        Debug.Log("Question 1 triggered--------------");
        answer_element = conditionCounter;
    }

    /// <summary>Hides Q1, shows Q2.</summary>
    public void Question2()
    {
        Q1.SetActive(false);
        Q2.SetActive(true);
        Debug.Log("Question TWOOO triggered--------------");
    }

    /// <summary>Hides Q2, shows Q3.</summary>
    public void Question3()
    {
        Q2.SetActive(false);
        Q3.SetActive(true);
        Debug.Log("Question THREE triggered--------------");
    }

    /// <summary>
    /// Finalizes answers for this trial:
    /// - Appends a line to mainData: "video_id,slider1,slider2,slider3"
    /// - Shows intermission panels at specific trial indices, or immediately starts next stage
    /// - Resets sliders and writes the CSV to disk
    /// </summary>
    public void Question4()
    {
        string mainLine = $"{trials[conditionCounter].video_id},{slider1.value},{slider2.value},{slider3.value}";
        mainData.Add(mainLine);

        Q3.SetActive(false);

        // Intermission logic at specific checkpoints
        if (((conditionCounter + 1) == 16) || ((conditionCounter + 1) == 28))
        {
            stop1.SetActive(true);
        }
        else if ((conditionCounter + 1) == 2)
        {
            start_study.SetActive(true);
        }
        else
        {
            // Reset sliders for next trial
            slider1.value = 0;
            slider2.value = 0;
            slider3.value = 0;

            // Signal we can proceed to the next trial
            startNextStage = true;

            // Persist questionnaire answers immediately
            writeCSV();
        }
    }

    public ToggleGroup toggleGroup, toggleGroup1;   // Not used in this code path
    private Toggle toggle, toggle1;                 // Not used
    public List<string> mainData = new List<string>(); // Accumulates questionnaire lines

    public void stop_screen1()
    {
        Debug.Log("Waiting time initialised.............");

        stop1.SetActive(false);
        slider1.value = 0; slider2.value = 0; slider3.value = 0;

        startNextStage = true;
        writeCSV();
    }

    public void stop_screen2()
    {
        Debug.Log("Waiting time initialised.............");

        start_study.SetActive(false);
        slider1.value = 0; slider2.value = 0; slider3.value = 0;

        startNextStage = true;
        writeCSV();
    }

    // =========================
    // XR Input logging (per-frame)
    // =========================

    private InputData _inputData;              // Not used, but left for future use
    private float _leftMaxScore = 0f;          // Not used
    private float _rightMaxScore = 0f;         // Not used

    public bool write_data = false;            // When true, UpdateInputDataDisplay writes csvData to disk and clears it

    public string filePath;                    // Per-trial CSV path (named with video_id)
    public string name;                        // Per-trial video_id
    private List<string> csvData = new List<string>(); // Per-frame XR rows

    string primaryButtonState = "False";       // (Unused remnants)

    // Previous-frame values to compute accelerations
    private Vector3 previousHmdVelocity = Vector3.zero;
    private Vector3 previousHmdAngularVelocity = Vector3.zero;
    private Vector3 previousLeftEyeVelocity = Vector3.zero;
    private Vector3 previousLeftEyeAngularVelocity = Vector3.zero;
    private Vector3 previousRightEyeVelocity = Vector3.zero;
    private Vector3 previousRightEyeAngularVelocity = Vector3.zero;
    private Vector3 previousCenterEyeVelocity = Vector3.zero;
    private Vector3 previousCenterEyeAngularVelocity = Vector3.zero;
    private float previousTime = 0.0f;

    /// <summary>
    /// Streams XR input data to an in-memory CSV buffer every frame:
    /// - Grabs OVR input for controllers (pos/rot/vel/angVel/accel) and button/touch states
    /// - Grabs XR InputDevices for HMD/eyes (pos/rot/vel/angVel) and derives accelerations
    /// - Emits one CSV-formatted line to csvData
    /// 
    /// When <see cref="write_data"/> is true:
    /// - Writes csvData to a CSV file named "&lt;video_id&gt;.csv" under Assets/
    /// - Resets write_data and clears the buffer
    /// </summary>
    private void UpdateInputDataDisplay()
    {
        // Timing info
        float currentTime = Time.time;
        float deltaTime = currentTime - previousTime;
        time2 = currentTime;
        string univ_timestamp = (time2 - time1).ToString();                  // Seconds since Start2()
        string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"); // Unused; left if needed

        // --- Controller transforms (OVRInput) ---
        Vector3 leftControllerPosition  = OVRInput.GetLocalControllerPosition(OVRInput.Controller.LTouch);
        Quaternion leftControllerRotation = OVRInput.GetLocalControllerRotation(OVRInput.Controller.LTouch);
        Vector3 rightControllerPosition = OVRInput.GetLocalControllerPosition(OVRInput.Controller.RTouch);
        Quaternion rightControllerRotation = OVRInput.GetLocalControllerRotation(OVRInput.Controller.RTouch);

        // --- Controller velocities & angular velocities ---
        Vector3 leftControllerVelocity  = OVRInput.GetLocalControllerVelocity(OVRInput.Controller.LTouch);
        Vector3 leftControllerAngularVelocity = OVRInput.GetLocalControllerAngularVelocity(OVRInput.Controller.LTouch);
        Vector3 rightControllerVelocity = OVRInput.GetLocalControllerVelocity(OVRInput.Controller.RTouch);
        Vector3 rightControllerAngularVelocity = OVRInput.GetLocalControllerAngularVelocity(OVRInput.Controller.RTouch);

        // --- Controller accelerations (direct from OVR) ---
        Vector3 leftControllerAcceleration = OVRInput.GetLocalControllerAcceleration(OVRInput.Controller.LTouch);
        Vector3 leftControllerAngularAcceleration = OVRInput.GetLocalControllerAngularAcceleration(OVRInput.Controller.LTouch);
        Vector3 rightControllerAcceleration = OVRInput.GetLocalControllerAcceleration(OVRInput.Controller.RTouch);
        Vector3 rightControllerAngularAcceleration = OVRInput.GetLocalControllerAngularAcceleration(OVRInput.Controller.RTouch);

        // --- HMD transforms/velocities (XR.InputDevices) ---
        InputDevice headDevice = InputDevices.GetDeviceAtXRNode(XRNode.Head);
        headDevice.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 hmdPosition);
        headDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion hmdRotation);
        headDevice.TryGetFeatureValue(CommonUsages.deviceVelocity, out Vector3 hmdVelocity);
        headDevice.TryGetFeatureValue(CommonUsages.deviceAngularVelocity, out Vector3 hmdAngularVelocity);

        // HMD acceleration derived by finite difference
        Vector3 hmdAcceleration = (hmdVelocity - previousHmdVelocity) / Mathf.Max(deltaTime, 1e-5f);
        Vector3 hmdAngularAcceleration = (hmdAngularVelocity - previousHmdAngularVelocity) / Mathf.Max(deltaTime, 1e-5f);

        // --- Left/Right/Center Eye transforms ---
        InputDevice leftEyeDevice = InputDevices.GetDeviceAtXRNode(XRNode.LeftEye);
        leftEyeDevice.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 leftEyePosition);
        leftEyeDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion leftEyeRotation);
        leftEyeDevice.TryGetFeatureValue(CommonUsages.deviceVelocity, out Vector3 leftEyeVelocity);
        leftEyeDevice.TryGetFeatureValue(CommonUsages.deviceAngularVelocity, out Vector3 leftEyeAngularVelocity);

        Vector3 leftEyeAcceleration = (leftEyeVelocity - previousLeftEyeVelocity) / Mathf.Max(deltaTime, 1e-5f);
        Vector3 leftEyeAngularAcceleration = (leftEyeAngularVelocity - previousLeftEyeAngularVelocity) / Mathf.Max(deltaTime, 1e-5f);

        InputDevice rightEyeDevice = InputDevices.GetDeviceAtXRNode(XRNode.RightEye);
        rightEyeDevice.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 rightEyePosition);
        rightEyeDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion rightEyeRotation);
        rightEyeDevice.TryGetFeatureValue(CommonUsages.deviceVelocity, out Vector3 rightEyeVelocity);
        rightEyeDevice.TryGetFeatureValue(CommonUsages.deviceAngularVelocity, out Vector3 rightEyeAngularVelocity);

        Vector3 rightEyeAcceleration = (rightEyeVelocity - previousRightEyeVelocity) / Mathf.Max(deltaTime, 1e-5f);
        Vector3 rightEyeAngularAcceleration = (rightEyeAngularVelocity - previousRightEyeAngularVelocity) / Mathf.Max(deltaTime, 1e-5f);

        InputDevice centerEyeDevice = InputDevices.GetDeviceAtXRNode(XRNode.CenterEye);
        centerEyeDevice.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 centerEyePosition);
        centerEyeDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion centerEyeRotation);
        centerEyeDevice.TryGetFeatureValue(CommonUsages.deviceVelocity, out Vector3 centerEyeVelocity);
        centerEyeDevice.TryGetFeatureValue(CommonUsages.deviceAngularVelocity, out Vector3 centerEyeAngularVelocity);

        Vector3 centerEyeAcceleration = (centerEyeVelocity - previousCenterEyeVelocity) / Mathf.Max(deltaTime, 1e-5f);
        Vector3 centerEyeAngularAcceleration = (centerEyeAngularVelocity - previousCenterEyeAngularVelocity) / Mathf.Max(deltaTime, 1e-5f);

        // --- Controller button/touch states (OVRInput) ---
        string primaryButtonStateLeft      = OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger,    OVRInput.Controller.LTouch) ? "True" : "False";
        string secondaryButtonStateLeft    = OVRInput.Get(OVRInput.Button.SecondaryIndexTrigger,  OVRInput.Controller.LTouch) ? "True" : "False";
        string primaryTouchStateLeft       = OVRInput.Get(OVRInput.Touch.PrimaryThumbRest,        OVRInput.Controller.LTouch) ? "True" : "False";
        string secondaryTouchStateLeft     = OVRInput.Get(OVRInput.Touch.SecondaryThumbRest,      OVRInput.Controller.LTouch) ? "True" : "False";

        string primaryButtonStateRight     = OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger,    OVRInput.Controller.RTouch) ? "True" : "False";
        string secondaryButtonStateRight   = OVRInput.Get(OVRInput.Button.SecondaryIndexTrigger,  OVRInput.Controller.RTouch) ? "True" : "False";
        string primaryTouchStateRight      = OVRInput.Get(OVRInput.Touch.PrimaryThumbRest,        OVRInput.Controller.RTouch) ? "True" : "False";
        string secondaryTouchStateRight    = OVRInput.Get(OVRInput.Touch.SecondaryThumbRest,      OVRInput.Controller.RTouch) ? "True" : "False";

        // --- Trigger/grip analogs (OVRInput) ---
        float triggerValueLeft = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.LTouch);
        float gripValueLeft    = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger,  OVRInput.Controller.LTouch);
        float triggerValueRight= OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.RTouch);
        float gripValueRight   = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger,  OVRInput.Controller.RTouch);

        // --- CSV line assembly (comma-separated numeric fields) ---
        string leftPosition  = $"{leftControllerPosition.x},{leftControllerPosition.y},{leftControllerPosition.z}";
        string leftRotation  = $"{leftControllerRotation.x},{leftControllerRotation.y},{leftControllerRotation.z},{leftControllerRotation.w}";
        string rightPosition = $"{rightControllerPosition.x},{rightControllerPosition.y},{rightControllerPosition.z}";
        string rightRotation = $"{rightControllerRotation.x},{rightControllerRotation.y},{rightControllerRotation.z},{rightControllerRotation.w}";

        string leftVelocity  = $"{leftControllerVelocity.x},{leftControllerVelocity.y},{leftControllerVelocity.z}";
        string leftAngularVelocity = $"{leftControllerAngularVelocity.x},{leftControllerAngularVelocity.y},{leftControllerAngularVelocity.z}";
        string rightVelocity = $"{rightControllerVelocity.x},{rightControllerVelocity.y},{rightControllerVelocity.z}";
        string rightAngularVelocity= $"{rightControllerAngularVelocity.x},{rightControllerAngularVelocity.y},{rightControllerAngularVelocity.z}";

        string leftAcceleration  = $"{leftControllerAcceleration.x},{leftControllerAcceleration.y},{leftControllerAcceleration.z}";
        string leftAngularAcceleration = $"{leftControllerAngularAcceleration.x},{leftControllerAngularAcceleration.y},{leftControllerAngularAcceleration.z}";
        string rightAcceleration = $"{rightControllerAcceleration.x},{rightControllerAcceleration.y},{rightControllerAcceleration.z}";
        string rightAngularAcceleration= $"{rightControllerAngularAcceleration.x},{rightControllerAngularAcceleration.y},{rightControllerAngularAcceleration.z}";

        string hmdPos   = $"{hmdPosition.x},{hmdPosition.y},{hmdPosition.z}";
        string hmdRot   = $"{hmdRotation.x},{hmdRotation.y},{hmdRotation.z},{hmdRotation.w}";
        string hmdVel   = $"{hmdVelocity.x},{hmdVelocity.y},{hmdVelocity.z}";
        string hmdAngVel= $"{hmdAngularVelocity.x},{hmdAngularVelocity.y},{hmdAngularVelocity.z}";
        string hmdAccel = $"{hmdAcceleration.x},{hmdAcceleration.y},{hmdAcceleration.z}";
        string hmdAngAccel= $"{hmdAngularAcceleration.x},{hmdAngularAcceleration.y},{hmdAngularAcceleration.z}";

        string leftEyePos   = $"{leftEyePosition.x},{leftEyePosition.y},{leftEyePosition.z}";
        string leftEyeRot   = $"{leftEyeRotation.x},{leftEyeRotation.y},{leftEyeRotation.z},{leftEyeRotation.w}";
        string leftEyeVel   = $"{leftEyeVelocity.x},{leftEyeVelocity.y},{leftEyeVelocity.z}";
        string leftEyeAngVel= $"{leftEyeAngularVelocity.x},{leftEyeAngularVelocity.y},{leftEyeAngularVelocity.z}";
        string leftEyeAccel = $"{leftEyeAcceleration.x},{leftEyeAcceleration.y},{leftEyeAcceleration.z}";
        string leftEyeAngAccel= $"{leftEyeAngularAcceleration.x},{leftEyeAngularAcceleration.y},{leftEyeAngularAcceleration.z}";

        string rightEyePos   = $"{rightEyePosition.x},{rightEyePosition.y},{rightEyePosition.z}";
        string rightEyeRot   = $"{rightEyeRotation.x},{rightEyeRotation.y},{rightEyeRotation.z},{rightEyeRotation.w}";
        string rightEyeVel   = $"{rightEyeVelocity.x},{rightEyeVelocity.y},{rightEyeVelocity.z}";
        string rightEyeAngVel= $"{rightEyeAngularVelocity.x},{rightEyeAngularVelocity.y},{rightEyeAngularVelocity.z}";
        string rightEyeAccel = $"{rightEyeAcceleration.x},{rightEyeAcceleration.y},{rightEyeAcceleration.z}";
        string rightEyeAngAccel= $"{rightEyeAngularAcceleration.x},{rightEyeAngularAcceleration.y},{rightEyeAngularAcceleration.z}";

        string centerEyePos   = $"{centerEyePosition.x},{centerEyePosition.y},{centerEyePosition.z}";
        string centerEyeRot   = $"{centerEyeRotation.x},{centerEyeRotation.y},{centerEyeRotation.z},{centerEyeRotation.w}";
        string centerEyeVel   = $"{centerEyeVelocity.x},{centerEyeVelocity.y},{centerEyeVelocity.z}";
        string centerEyeAngVel= $"{centerEyeAngularVelocity.x},{centerEyeAngularVelocity.y},{centerEyeAngularVelocity.z}";
        string centerEyeAccel = $"{centerEyeAcceleration.x},{centerEyeAcceleration.y},{centerEyeAcceleration.z}";
        string centerEyeAngAccel= $"{centerEyeAngularAcceleration.x},{centerEyeAngularAcceleration.y},{centerEyeAngularAcceleration.z}";

        // Full CSV row
        string csvLine =
            $"{univ_timestamp}," +
            $"{leftPosition},{leftRotation},{leftVelocity},{leftAngularVelocity},{leftAcceleration},{leftAngularAcceleration}," +
            $"{rightPosition},{rightRotation},{rightVelocity},{rightAngularVelocity},{rightAcceleration},{rightAngularAcceleration}," +
            $"{hmdPos},{hmdRot},{hmdVel},{hmdAngVel},{hmdAccel},{hmdAngAccel}," +
            $"{leftEyePos},{leftEyeRot},{leftEyeVel},{leftEyeAngVel},{leftEyeAccel},{leftEyeAngAccel}," +
            $"{rightEyePos},{rightEyeRot},{rightEyeVel},{rightEyeAngVel},{rightEyeAccel},{rightEyeAngAccel}," +
            $"{centerEyePos},{centerEyeRot},{centerEyeVel},{centerEyeAngVel},{centerEyeAccel},{centerEyeAngAccel}," +
            $"{primaryButtonStateLeft},{secondaryButtonStateLeft},{primaryTouchStateLeft},{secondaryTouchStateLeft}," +
            $"{primaryButtonStateRight},{secondaryButtonStateRight},{primaryTouchStateRight},{secondaryTouchStateRight}," +
            $"{triggerValueLeft},{gripValueLeft},{triggerValueRight},{gripValueRight}";

        // Accumulate for this trial
        csvData.Add(csvLine);

        // Carry previous velocities for acceleration next frame
        previousHmdVelocity               = hmdVelocity;
        previousHmdAngularVelocity        = hmdAngularVelocity;
        previousLeftEyeVelocity           = leftEyeVelocity;
        previousLeftEyeAngularVelocity    = leftEyeAngularVelocity;
        previousRightEyeVelocity          = rightEyeVelocity;
        previousRightEyeAngularVelocity   = rightEyeAngularVelocity;
        previousCenterEyeVelocity         = centerEyeVelocity;
        previousCenterEyeAngularVelocity  = centerEyeAngularVelocity;
        previousTime                      = currentTime;

        // If flagged, flush this trial's buffered XR lines to <video_id>.csv and clear
        if (write_data == true)
        {
            name = trials[conditionCounter].video_id; // File base name
            filePath = Application.dataPath + "/" + name + ".csv";

            using (TextWriter tw = new StreamWriter(filePath, false))
            {
                // Header (keep in sync with csvLine)
                tw.WriteLine("Timestamp,LeftPositionX,LeftPositionY,LeftPositionZ,LeftRotationX,LeftRotationY,LeftRotationZ,LeftRotationW,LeftVelocityX,LeftVelocityY,LeftVelocityZ,LeftAngularVelocityX,LeftAngularVelocityY,LeftAngularVelocityZ,LeftAccelerationX,LeftAccelerationY,LeftAccelerationZ,LeftAngularAccelerationX,LeftAngularAccelerationY,LeftAngularAccelerationZ,RightPositionX,RightPositionY,RightPositionZ,RightRotationX,RightRotationY,RightRotationZ,RightRotationW,RightVelocityX,RightVelocityY,RightVelocityZ,RightAngularVelocityX,RightAngularVelocityY,RightAngularVelocityZ,RightAccelerationX,RightAccelerationY,RightAccelerationZ,RightAngularAccelerationX,RightAngularAccelerationY,RightAngularAccelerationZ,HMDPositionX,HMDPositionY,HMDPositionZ,HMDRotationX,HMDRotationY,HMDRotationZ,HMDRotationW,HMDVelocityX,HMDVelocityY,HMDVelocityZ,HMDAngularVelocityX,HMDAngularVelocityY,HMDAngularVelocityZ,HMDAccelerationX,HMDAccelerationY,HMDAccelerationZ,HMDAngularAccelerationX,HMDAngularAccelerationY,HMDAngularAccelerationZ,LeftEyePositionX,LeftEyePositionY,LeftEyePositionZ,LeftEyeRotationX,LeftEyeRotationY,LeftEyeRotationZ,LeftEyeRotationW,LeftEyeVelocityX,LeftEyeVelocityY,LeftEyeVelocityZ,LeftEyeAngularVelocityX,LeftEyeAngularVelocityY,LeftEyeAngularVelocityZ,LeftEyeAccelerationX,LeftEyeAccelerationY,LeftEyeAccelerationZ,LeftEyeAngularAccelerationX,LeftEyeAngularAccelerationY,LeftEyeAngularAccelerationZ,RightEyePositionX,RightEyePositionY,RightEyePositionZ,RightEyeRotationX,RightEyeRotationY,RightEyeRotationZ,RightEyeRotationW,RightEyeVelocityX,RightEyeVelocityY,RightEyeVelocityZ,RightEyeAngularVelocityX,RightEyeAngularVelocityY,RightEyeAngularVelocityZ,RightEyeAccelerationX,RightEyeAccelerationY,RightEyeAccelerationZ,RightEyeAngularAccelerationX,RightEyeAngularAccelerationY,RightEyeAngularAccelerationZ,CenterEyePositionX,CenterEyePositionY,CenterEyePositionZ,CenterEyeRotationX,CenterEyeRotationY,CenterEyeRotationZ,CenterEyeRotationW,CenterEyeVelocityX,CenterEyeVelocityY,CenterEyeVelocityZ,CenterEyeAngularVelocityX,CenterEyeAngularVelocityY,CenterEyeAngularVelocityZ,CenterEyeAccelerationX,CenterEyeAccelerationY,CenterEyeAccelerationZ,CenterEyeAngularAccelerationX,CenterEyeAngularAccelerationY,CenterEyeAngularAccelerationZ,PrimaryButtonStateLeft,SecondaryButtonStateLeft,PrimaryTouchStateLeft,SecondaryTouchStateLeft,PrimaryButtonStateRight,SecondaryButtonStateRight,PrimaryTouchStateRight,SecondaryTouchStateRight,TriggerValueLeft,GripValueLeft,TriggerValueRight,GripValueRight");

                // Data rows
                foreach (string line in csvData)
                {
                    tw.WriteLine(line);
                }
            }

            // Reset write flag and buffer for next trial
            write_data = false;
            csvData.Clear();
        }
    }
}
