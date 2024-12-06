using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using System;
using UnityEngine.XR;
using System.IO;
using UnityEditor.Recorder;
using UnityEditor.Recorder.Input;
using System.Linq;

[System.Serializable]
public class Trial {
    public int no;
    public string scenario;
    public string video_id;     //USE THIS FOR identification
    public int yielding;
    public int eHMIOn;
    public int distPed;
    public int p1;
    public int p2;
    public int camera;
    public int group;
    public int video_length;
    public string sound_clip_name;
}

public class ConditionController : MonoBehaviour
{

    public string writeFileName = "a";       //write the name of the file for storing the data into----------a for standard
    public string writeFilePath = "";           //where the file will be saved

    //------------for writing

    private float initialHeight;
    public bool conditionFinished = false;
    LightStripBumper lightStripBumper;      // script
    public GameObject LEDBumperObject;
    public GameObject tracker;
    public GameObject progress;
    public GameObject projection;

    CarMovement carMovementScript;
    PlayFabController playfabScript;
    public int conditionCounter = 0;
    public int numberConditions = 0;

    public int eHMIOn = 0;   // 0=no 1=slowly-pulsing light band
    public int yielding = 0; // 0=yes for P1 1=yes for P2 2=no
    public int distPed = 0;  // distance between P1 and P2 distances [2 .. +2 ..  20].
    public int p1 = 0;       // presence of Pedestrian 1
    public int p2 = 0;       // presence of Pedestrian 2
    public int camera = 0;   // location of camera
    public int duration = 0;

    public GameObject demoWelcomeCanvas;
    public GameObject demoWalkCanvas;
    public GameObject demoInfoCanvas1;
    public GameObject demoInfoCanvas2;

    public GameObject trialWalkCanvas;
    public GameObject trialDemoCanvas;
    public GameObject trialStartCanvas;
    public GameObject trialEndCanvas;
    public GameObject ExperimentEndCanvas;

    public Text demoTitle;
    public Text demoText;
    public Text trialTitle;

    // todo: cleanup
    public GameObject WillingnessToCross;
    public GameObject reticle;
    public GameObject CountDown;
    public bool preview = false;
    public bool trial = false;

    public AudioSource buttonSound;

    public Trial[] trials; // description of trials based on mapping

    public GameObject p1_object;  // object of P1
    public GameObject p2_object;  // object of P2
    public GameObject camera_object;  // camera object
    public GameObject black_canvas;  // canvas to be shown as black screen
    public AudioSource carAudioSource; // Reference to the car's AudioSource
    private AudioClip currentClip;

    public void Start()
    {
        // Import trial data
        string filePath = Application.dataPath + "/../../mapping.csv";
        ShuffleCSVFile();

        writeFilePath = Application.dataPath +"/" +  writeFileName + DateTime.Now.ToString("yyyyMMdd_HHmmss") +  ".csv";            //the path to stroe the files with the given filename(change it for unique files)

        if (File.Exists(writeFilePath))
        {
            Debug.Log("File already exists, deleting...");
            File.Delete(writeFilePath);
        }

        Debug.Log("Start");
        
        string text = File.ReadAllText(filePath);
        trials = CSVSerializer.Deserialize<Trial>(text);
        numberConditions = trials.Length; // set number of conditions
        Debug.Log("Number of conditions: " + numberConditions);
        StartCoroutine(ActivatorVR("cardboard"));
        buttonSound = GetComponent<AudioSource>();

        Start2();

    }

    public void ShuffleCSVFile()
    {
        string filePath = Application.dataPath + "/../../mapping.csv";

        if (!File.Exists(filePath))
        {
            Debug.LogError("CSV file not found at: " + filePath);
            return;
        }

        // Read the CSV lines into a list
        var lines = File.ReadAllLines(filePath).ToList();

        if (lines.Count <= 2)
        {
            Debug.LogError("The CSV file does not have enough rows to shuffle.");
            return;
        }

        // Separate the header (first line) and the data rows
        var header = lines[0];  // Keep the header as is
        var firstRow = lines[1];  // The first data row (you mentioned this should not be shuffled)
        var data = lines.Skip(2).ToList();  // All rows after the first two rows

        // Shuffle the remaining rows using System.Random (explicitly qualified)
        System.Random rand = new System.Random(42);  // Set seed for reproducibility
        var shuffledData = data.OrderBy(x => rand.Next()).ToList();

        // Concatenate the header, first row, and shuffled data
        var result = new List<string> { header, firstRow }.Concat(shuffledData).ToList();

        // Save the shuffled result back to the CSV file
        File.WriteAllLines(filePath, result);

        Debug.Log("CSV rows have been shuffled and saved to: " + filePath);
    }


    public IEnumerator ActivatorVR(string YESVR)
    {

       // XRSettings.LoadDeviceByName(YESVR);
        yield return null;
        //XRSettings.enabled = true;
    }
    public IEnumerator DectivatorVR(string NOVR)
    {
      //  XRSettings.LoadDeviceByName(NOVR);
        yield return null;
       // XRSettings.enabled = false;
    }

    public float time1, time2 = 0f;
    Vector3 initialPositionP1 = new Vector3(105.792694f,-3.31599998f,3.44000006f);
    Vector3 initialPositionP2 = new Vector3(104.082703f,-3.31599998f,3.44000006f);
    void ResetPositions() {
        p1_object.transform.position = initialPositionP1;
        p2_object.transform.position = initialPositionP2;
    }

    public GameObject player1, player2;
    void Start2()
    {
        ResetPositions();
        time1 = Time.time;
        carMovementScript = GameObject.Find("CarMovement").GetComponent<CarMovement>();
        playfabScript = GameObject.Find("PlayFabController").GetComponent<PlayFabController>();
        LEDBumperObject.SetActive(true);            // Turn on LED bumper
        tracker.SetActive(false);                   // Switch off tracker
        progress.SetActive(false);                  // Switch off progressbar
        projection.SetActive(false);                // Switch off projection

        // Set variables for trial
        eHMIOn = trials[conditionCounter].eHMIOn;
        yielding = trials[conditionCounter].yielding;
        distPed = trials[conditionCounter].distPed;
        p1 = trials[conditionCounter].p1;
        p2 = trials[conditionCounter].p2;
        camera = trials[conditionCounter].camera;

        duration = trials[conditionCounter].video_length;
        string soundName = trials[conditionCounter].sound_clip_name;

        Debug.Log(conditionCounter +  ":: eHMIOn= " + eHMIOn +  " yielding= " + yielding +  " distPed= " + distPed +
          " p1= " + p1 +  " p2= " + p2 + " camera= " + camera + " sound name= " + soundName);

        // Make p1 present or not
        if (p1 == 0) {
            p1_object.SetActive(false);
            Debug.Log("P1 disabled");
        } else {
            p1_object.SetActive(true);
            Debug.Log("P1 enabled");
        }

        // Make p2 present or not
        if (p2 == 0) {
            p2_object.SetActive(false);
            Debug.Log("P2 disabled");
        } else {
            p2_object.SetActive(true);
            Debug.Log("P2 enabled");
        }

        // Distance between pedestrians
        // position of P1=(21.3, -3.316, -3.98272)
        Debug.Log("Case start - P1: " + p1_object.transform.position + ", P2: " + p2_object.transform.position);
        float deltaDist = 2f * distPed; // change in x coordinate
        float adjustment = (5.0f - distPed) * 2;
        if (distPed != 0) {
            p2_object.transform.position = new Vector3(p1_object.transform.position.x - deltaDist,
                                                       p1_object.transform.position.y,
                                                       p1_object.transform.position.z);

            Debug.Log("adjustment" + adjustment);
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

        // Camera position
        Vector3 posCameraP1 = new Vector3(105.54f, -1.717f, 3.6f);  // position of camera for 
        Vector3 rotCameraP1 = new Vector3(0f, 0f, 0f);   // rotation of camera for P1
        Vector3 rotCameraP2 = new Vector3(0f, 0f, 0f);   // rotation of camera for P2
        Vector3 posCamera3rd = new Vector3(108.53f, -0.47f, -2.68f);  // position of camera for 3rd person view
        Vector3 rotCamera3rd = new Vector3(0f, -49.995f, 0f);   // rotation of camera for 3rd person view
        Vector3 targetCameraPos = new Vector3();  // target for moving camera
        float transitionDuration = 0.0f;;  // duration of movement of camera
        if (camera == 0) {
            camera_object.transform.position = posCameraP1;
            camera_object.transform.eulerAngles = rotCameraP1;
            camera_object.transform.position = new Vector3(posCameraP1.x - ((5.0f - (distPed))*2),  // take into account movement of P2
                                                           posCameraP1.y,
                                                           posCameraP1.z);
            Debug.Log("Camera set to head of P1.");
        } else if (camera == 1) {
            // First update: Intermediate position considering deltaDist
            Vector3 intermediateCameraPos = new Vector3(posCameraP1.x - deltaDist, 
                                                posCameraP1.y,
                                                posCameraP1.z);

            // Apply the first update
            camera_object.transform.position = intermediateCameraPos;
            camera_object.transform.eulerAngles = rotCameraP1;

            // Second update: Further adjust the camera position
            Vector3 finalCameraPos = new Vector3(posCameraP1.x - 10.0f, 
                                            posCameraP1.y,
                                            posCameraP1.z);

            // Apply the second update
            camera_object.transform.position = finalCameraPos;
            Debug.Log("Camera set to head of P2.");

        } else if (camera == 2) {
            camera_object.transform.position = posCamera3rd;
            camera_object.transform.eulerAngles = rotCameraP2;
            Debug.Log("Camera set to 3rd person view.");
        } else if (camera == 3) {
            camera_object.transform.position = new Vector3(posCameraP1.x - deltaDist,  // take into account movement of P2
                                                           posCameraP1.y,
                                                           posCameraP1.z);
            camera_object.transform.eulerAngles = rotCameraP2;
            targetCameraPos = posCameraP1;
            // todo: dynamic value for time
            transitionDuration = 0.5f * deltaDist;
            Debug.Log("Camera set to P1 with going away from P2.");
        } else if (camera == 4) {
            camera_object.transform.position = posCameraP1;
            camera_object.transform.eulerAngles = rotCameraP1;
            targetCameraPos = new Vector3(posCameraP1.x - deltaDist,  // take into account movement of P2
                                          posCameraP1.y,
                                          posCameraP1.z);
            transitionDuration = 0.5f * deltaDist;  // no camera movement needed
            Debug.Log("Camera set to P2 with going away from P1.");
        } else {
            Debug.Log("Wrong value for camera given.");
        }

        LoadAndPlaySound(soundName, duration);

        // Show black screen for 1 s
        StartCoroutine(BlackScreen(1f));
        // Start trial
        TrialStart(targetCameraPos, transitionDuration);
        // End recording video

        startNextStage = false;                             //-------next stage stops and waits for the input

        StartCoroutine(UI_duration(duration));
        initialHeight = Camera.main.transform.position.y;
        // Question1();
    }

    void LoadAndPlaySound(string soundName, int videoDuration)
    {
        if (carAudioSource == null)
        {
            Debug.LogError("AudioSource not assigned to car.");
            return;
        }

        string soundPath = "Sound/" + soundName; // Correct path relative to Resources
        currentClip = Resources.Load<AudioClip>(soundPath);
        if (currentClip == null)
        {
            Debug.LogError("Sound file not found: " + soundPath);
            return;
        }

        carAudioSource.clip = currentClip;
        carAudioSource.Play();
        Debug.Log("Playing sound: " + soundName);

        StartCoroutine(StopSoundAfterDuration(videoDuration));
    }

    IEnumerator StopSoundAfterDuration(int videoDuration)
    {
        yield return new WaitForSeconds(videoDuration/1000);
        carAudioSource.Stop();
        Debug.Log("Sound stopped");
    }

    IEnumerator UI_duration(int time_duration)
    {
        yield return new WaitForSeconds(time_duration/1000);
        Question1();
    }

    // Show black screen for 1 second
    IEnumerator BlackScreen(float t)
    {
        black_canvas.GetComponent<Image>().color = new Color(0, 0, 0, 255);
        yield return new WaitForSeconds(t);
        black_canvas.GetComponent<Image>().color = new Color(0, 0, 0, 0);
    }

    private void FixedUpdate()
    {
        Vector3 currentPosition = Camera.main.transform.position;
        if (Mathf.Abs(currentPosition.y - initialHeight) > 0.01f)  // Adjust the threshold as needed
        {
            currentPosition.y = initialHeight;
            Camera.main.transform.position = currentPosition;
        }


                                    if (startNextStage == true)//writes and saves a file and starts the new cycle of data input
                                        { write_data = true; }
        // Update input data display
        UpdateInputDataDisplay();

        if (carMovementScript != null) {
            if (carMovementScript.conditionFinished && startNextStage == true)            //ONE experiment is done    //-------------------assuming this starts the next stage
            {
                // Experiment is finished
                if (conditionCounter == numberConditions - 1) {
                    Debug.Log("Experiment finished");
                    SceneManager.LoadScene("EndMenu");
                    Application.Quit(); // quit
                }
                Debug.Log("FixedUpdate::trial end");
                WillingnessToCross.SetActive(false);
                reticle.SetActive(true);
                carMovementScript.conditionFinished = false;
                trial = false;

                conditionCounter = conditionCounter + 1;
                trialEndCanvas.SetActive(false);
                StartCoroutine(ActivatorVR("none"));
                if (startNextStage == true)
                Start2();
            }
        }
    }

    // UI DEMO
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

    IEnumerator WalkForward()
    {
        Debug.Log("WalkForward");
        yield return new WaitForSecondsRealtime(0.2f);
        GameObject.Find("CameraHolder").GetComponent<MoveCamera>().StartWalk = true;
        yield return new WaitForSecondsRealtime(3.0f);
    }

    // UI TRIALS
    void TrialStart(Vector3 targetCameraPos, float transitionDuration)
    {
        Debug.Log("TrialStart");
        TrialCanvas3(targetCameraPos, transitionDuration);
        //trialWalkCanvas.SetActive(true);
    }
    public void TrialCanvas1()
    {
        Debug.Log("TrialCanvas1");
        trialWalkCanvas.SetActive(false);
        StartCoroutine(WalkForward());
        trialDemoCanvas.SetActive(true);
    }
    public void TrialCanvas2()                  // Start preview
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
    public void TrialCanvas3(Vector3 targetCameraPos, float transitionDuration)                  // Start trial
    {
        Debug.Log("TrialCanvas3");
        // trialStartCanvas.SetActive(false);
        carMovementScript.AudioBeep.Play();
        // move camera
        if (transitionDuration > 0) {
            StartCoroutine(MoveCamera(targetCameraPos, transitionDuration));
        }
        trial = true;
        carMovementScript.StartCar();

        // StartCoroutine(CountDownTrial());
    }

    // move camera between two points
    IEnumerator MoveCamera(Vector3 targetCameraPos, float transitionDuration) {
        Debug.Log("Moving camera");
        yield return new WaitForSeconds(1f);
        Debug.Log(camera_object.transform.position);
        Debug.Log(targetCameraPos);
        float t = 0.0f;
        while (t < 1.0f)
        {
            t += Time.deltaTime * (Time.timeScale/transitionDuration);
            camera_object.transform.position = Vector3.Lerp(camera_object.transform.position, // move from
                                                            targetCameraPos,  // move to
                                                            t);  // in amount of time
            Debug.Log(t);
            yield return 0;
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
        // Set next condition
        // PlayerPrefs.SetInt("Condition Counter", conditionCounter + 1);
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
    //writing function

    public void writeCSV()
    {

        Debug.Log("----------file writing triggered");
                TextWriter tw = new StreamWriter(writeFilePath, false);
                tw.WriteLine("Video ID, Answer1, Answer2, Answer3");        //headings
                tw.Close();


        File.WriteAllLines(writeFilePath, mainData);

        //mainData.Clear();            //emoptyy the file

        tw.Close();
    }

    public int answer_element = 0;
    public GameObject Q1, Q2, Q3;       // the panels
    public GameObject stop1, start_study;
    public Slider slider1;       // the first Q slider
    public Slider slider2;       // the second Q slider
    public Slider slider3;       // the second Q slider
    public bool startNextStage = false;
    public void Question1()     //call this after the end of every frame
    {
        Q1.SetActive(true);
        // Time.timeScale = 0f; // Pause the game by setting time scale to 0

        Debug.Log("Question 1 triggered--------------");
        //take the experiment number and put it as an array number
        answer_element = conditionCounter;

    }

    public void Question2()     //call this on the press of next button on q1
    {

        Q1.SetActive(false);

        Q2.SetActive(true);

        Debug.Log("Question TWOOO triggered--------------");

    }
    public void Question3() //call this on the press of next button on q2
    {

        Q2.SetActive(false);

        Q3.SetActive(true);

        Debug.Log("Question THREE triggered--------------");

    }
    public void Question4() //call this on the press of next button on q2
    {

        string mainLine = $"{trials[conditionCounter].video_id},{slider1.value},{slider2.value},{slider3.value}";
        mainData.Add(mainLine);

        Q3.SetActive(false);

        // if (((conditionCounter + 1) == 16) || ((conditionCounter + 1) == 28))
        // {
        //     stop1.SetActive(true);
        // }
        if ((conditionCounter % 1) == 0)
        {
            start_study.SetActive(true);
        }
        else{
            //resetting values
            slider1.value = 0;
            slider2.value = 0;
            slider3.value = 0;

            startNextStage = true;

            writeCSV();
        }

    }
    public ToggleGroup toggleGroup, toggleGroup1;
    private Toggle toggle, toggle1;
    public List<string> mainData = new List<string>();

    public void stop_screen1()
    {
        Debug.Log("Waiting time initialised.............");

        stop1.SetActive(false);
        //resetting values
        slider1.value = 0;
        slider2.value = 0;
        slider3.value = 0;

        //Time.timeScale = 1f; // Play the game by setting time scale to 1

        startNextStage = true;

        writeCSV();

    }
    public void stop_screen2()
    {
        Debug.Log("Waiting time initialised.............");

        start_study.SetActive(false);
        //resetting values
        slider1.value = 0;
        slider2.value = 0;
        slider3.value = 0;

        //Time.timeScale = 1f; // Play the game by setting time scale to 1

        startNextStage = true;

        writeCSV();

    }
    private InputData _inputData;
    private float _leftMaxScore = 0f;
    private float _rightMaxScore = 0f;

    public bool write_data = false;     //make this true whenever writing is required

    public string filePath;
    public string name;
    private List<string> csvData = new List<string>();

    string primaryButtonState = "False";
    string secondaryButtonState = "False";


// Declare these variables to keep track of previous values
private Vector3 previousHmdVelocity = Vector3.zero;
private Vector3 previousHmdAngularVelocity = Vector3.zero;
private Vector3 previousLeftEyeVelocity = Vector3.zero;
private Vector3 previousLeftEyeAngularVelocity = Vector3.zero;
private Vector3 previousRightEyeVelocity = Vector3.zero;
private Vector3 previousRightEyeAngularVelocity = Vector3.zero;
private Vector3 previousCenterEyeVelocity = Vector3.zero;
private Vector3 previousCenterEyeAngularVelocity = Vector3.zero;
private float previousTime = 0.0f;

private void UpdateInputDataDisplay()
{
    float currentTime = Time.time;
    float deltaTime = currentTime - previousTime;
    time2 = currentTime;
    string univ_timestamp = (time2 - time1).ToString();
    string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");

    // Get the device positions and rotations for controllers
    Vector3 leftControllerPosition = OVRInput.GetLocalControllerPosition(OVRInput.Controller.LTouch);
    Quaternion leftControllerRotation = OVRInput.GetLocalControllerRotation(OVRInput.Controller.LTouch);
    Vector3 rightControllerPosition = OVRInput.GetLocalControllerPosition(OVRInput.Controller.RTouch);
    Quaternion rightControllerRotation = OVRInput.GetLocalControllerRotation(OVRInput.Controller.RTouch);

    // Get the device velocities and angular velocities for controllers
    Vector3 leftControllerVelocity = OVRInput.GetLocalControllerVelocity(OVRInput.Controller.LTouch);
    Vector3 leftControllerAngularVelocity = OVRInput.GetLocalControllerAngularVelocity(OVRInput.Controller.LTouch);
    Vector3 rightControllerVelocity = OVRInput.GetLocalControllerVelocity(OVRInput.Controller.RTouch);
    Vector3 rightControllerAngularVelocity = OVRInput.GetLocalControllerAngularVelocity(OVRInput.Controller.RTouch);

    // Get the device accelerations and angular accelerations for controllers
    Vector3 leftControllerAcceleration = OVRInput.GetLocalControllerAcceleration(OVRInput.Controller.LTouch);
    Vector3 leftControllerAngularAcceleration = OVRInput.GetLocalControllerAngularAcceleration(OVRInput.Controller.LTouch);
    Vector3 rightControllerAcceleration = OVRInput.GetLocalControllerAcceleration(OVRInput.Controller.RTouch);
    Vector3 rightControllerAngularAcceleration = OVRInput.GetLocalControllerAngularAcceleration(OVRInput.Controller.RTouch);

    // Get the device position, rotation, velocity, and angular velocity for HMD
    InputDevice headDevice = InputDevices.GetDeviceAtXRNode(XRNode.Head);
    headDevice.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 hmdPosition);
    headDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion hmdRotation);
    headDevice.TryGetFeatureValue(CommonUsages.deviceVelocity, out Vector3 hmdVelocity);
    headDevice.TryGetFeatureValue(CommonUsages.deviceAngularVelocity, out Vector3 hmdAngularVelocity);

    // Compute acceleration and angular acceleration for HMD
    Vector3 hmdAcceleration = (hmdVelocity - previousHmdVelocity) / deltaTime;
    Vector3 hmdAngularAcceleration = (hmdAngularVelocity - previousHmdAngularVelocity) / deltaTime;

    // Get the device position, rotation, velocity, and angular velocity for left eye
    InputDevice leftEyeDevice = InputDevices.GetDeviceAtXRNode(XRNode.LeftEye);
    leftEyeDevice.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 leftEyePosition);
    leftEyeDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion leftEyeRotation);
    leftEyeDevice.TryGetFeatureValue(CommonUsages.deviceVelocity, out Vector3 leftEyeVelocity);
    leftEyeDevice.TryGetFeatureValue(CommonUsages.deviceAngularVelocity, out Vector3 leftEyeAngularVelocity);

    // Compute acceleration and angular acceleration for left eye
    Vector3 leftEyeAcceleration = (leftEyeVelocity - previousLeftEyeVelocity) / deltaTime;
    Vector3 leftEyeAngularAcceleration = (leftEyeAngularVelocity - previousLeftEyeAngularVelocity) / deltaTime;

    // Get the device position, rotation, velocity, and angular velocity for right eye
    InputDevice rightEyeDevice = InputDevices.GetDeviceAtXRNode(XRNode.RightEye);
    rightEyeDevice.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 rightEyePosition);
    rightEyeDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion rightEyeRotation);
    rightEyeDevice.TryGetFeatureValue(CommonUsages.deviceVelocity, out Vector3 rightEyeVelocity);
    rightEyeDevice.TryGetFeatureValue(CommonUsages.deviceAngularVelocity, out Vector3 rightEyeAngularVelocity);

    // Compute acceleration and angular acceleration for right eye
    Vector3 rightEyeAcceleration = (rightEyeVelocity - previousRightEyeVelocity) / deltaTime;
    Vector3 rightEyeAngularAcceleration = (rightEyeAngularVelocity - previousRightEyeAngularVelocity) / deltaTime;

    // Get the device position, rotation, velocity, and angular velocity for center eye
    InputDevice centerEyeDevice = InputDevices.GetDeviceAtXRNode(XRNode.CenterEye);
    centerEyeDevice.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 centerEyePosition);
    centerEyeDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion centerEyeRotation);
    centerEyeDevice.TryGetFeatureValue(CommonUsages.deviceVelocity, out Vector3 centerEyeVelocity);
    centerEyeDevice.TryGetFeatureValue(CommonUsages.deviceAngularVelocity, out Vector3 centerEyeAngularVelocity);

    // Compute acceleration and angular acceleration for center eye
    Vector3 centerEyeAcceleration = (centerEyeVelocity - previousCenterEyeVelocity) / deltaTime;
    Vector3 centerEyeAngularAcceleration = (centerEyeAngularVelocity - previousCenterEyeAngularVelocity) / deltaTime;

    // Convert vectors and quaternions to string
    string leftPosition = $"{leftControllerPosition.x},{leftControllerPosition.y},{leftControllerPosition.z}";
    string leftRotation = $"{leftControllerRotation.x},{leftControllerRotation.y},{leftControllerRotation.z},{leftControllerRotation.w}";
    string rightPosition = $"{rightControllerPosition.x},{rightControllerPosition.y},{rightControllerPosition.z}";
    string rightRotation = $"{rightControllerRotation.x},{rightControllerRotation.y},{rightControllerRotation.z},{rightControllerRotation.w}";

    string leftVelocity = $"{leftControllerVelocity.x},{leftControllerVelocity.y},{leftControllerVelocity.z}";
    string leftAngularVelocity = $"{leftControllerAngularVelocity.x},{leftControllerAngularVelocity.y},{leftControllerAngularVelocity.z}";
    string rightVelocity = $"{rightControllerVelocity.x},{rightControllerVelocity.y},{rightControllerVelocity.z}";
    string rightAngularVelocity = $"{rightControllerAngularVelocity.x},{rightControllerAngularVelocity.y},{rightControllerAngularVelocity.z}";

    string leftAcceleration = $"{leftControllerAcceleration.x},{leftControllerAcceleration.y},{leftControllerAcceleration.z}";
    string leftAngularAcceleration = $"{leftControllerAngularAcceleration.x},{leftControllerAngularAcceleration.y},{leftControllerAngularAcceleration.z}";
    string rightAcceleration = $"{rightControllerAcceleration.x},{rightControllerAcceleration.y},{rightControllerAcceleration.z}";
    string rightAngularAcceleration = $"{rightControllerAngularAcceleration.x},{rightControllerAngularAcceleration.y},{rightControllerAngularAcceleration.z}";

    string hmdPos = $"{hmdPosition.x},{hmdPosition.y},{hmdPosition.z}";
    string hmdRot = $"{hmdRotation.x},{hmdRotation.y},{hmdRotation.z},{hmdRotation.w}";
    string hmdVel = $"{hmdVelocity.x},{hmdVelocity.y},{hmdVelocity.z}";
    string hmdAngVel = $"{hmdAngularVelocity.x},{hmdAngularVelocity.y},{hmdAngularVelocity.z}";
    string hmdAccel = $"{hmdAcceleration.x},{hmdAcceleration.y},{hmdAcceleration.z}";
    string hmdAngAccel = $"{hmdAngularAcceleration.x},{hmdAngularAcceleration.y},{hmdAngularAcceleration.z}";

    string leftEyePos = $"{leftEyePosition.x},{leftEyePosition.y},{leftEyePosition.z}";
    string leftEyeRot = $"{leftEyeRotation.x},{leftEyeRotation.y},{leftEyeRotation.z},{leftEyeRotation.w}";
    string leftEyeVel = $"{leftEyeVelocity.x},{leftEyeVelocity.y},{leftEyeVelocity.z}";
    string leftEyeAngVel = $"{leftEyeAngularVelocity.x},{leftEyeAngularVelocity.y},{leftEyeAngularVelocity.z}";
    string leftEyeAccel = $"{leftEyeAcceleration.x},{leftEyeAcceleration.y},{leftEyeAcceleration.z}";
    string leftEyeAngAccel = $"{leftEyeAngularAcceleration.x},{leftEyeAngularAcceleration.y},{leftEyeAngularAcceleration.z}";

    string rightEyePos = $"{rightEyePosition.x},{rightEyePosition.y},{rightEyePosition.z}";
    string rightEyeRot = $"{rightEyeRotation.x},{rightEyeRotation.y},{rightEyeRotation.z},{rightEyeRotation.w}";
    string rightEyeVel = $"{rightEyeVelocity.x},{rightEyeVelocity.y},{rightEyeVelocity.z}";
    string rightEyeAngVel = $"{rightEyeAngularVelocity.x},{rightEyeAngularVelocity.y},{rightEyeAngularVelocity.z}";
    string rightEyeAccel = $"{rightEyeAcceleration.x},{rightEyeAcceleration.y},{rightEyeAcceleration.z}";
    string rightEyeAngAccel = $"{rightEyeAngularAcceleration.x},{rightEyeAngularAcceleration.y},{rightEyeAngularAcceleration.z}";

    string centerEyePos = $"{centerEyePosition.x},{centerEyePosition.y},{centerEyePosition.z}";
    string centerEyeRot = $"{centerEyeRotation.x},{centerEyeRotation.y},{centerEyeRotation.z},{centerEyeRotation.w}";
    string centerEyeVel = $"{centerEyeVelocity.x},{centerEyeVelocity.y},{centerEyeVelocity.z}";
    string centerEyeAngVel = $"{centerEyeAngularVelocity.x},{centerEyeAngularVelocity.y},{centerEyeAngularVelocity.z}";
    string centerEyeAccel = $"{centerEyeAcceleration.x},{centerEyeAcceleration.y},{centerEyeAcceleration.z}";
    string centerEyeAngAccel = $"{centerEyeAngularAcceleration.x},{centerEyeAngularAcceleration.y},{centerEyeAngularAcceleration.z}";

    // Button and touch states for both controllers
    string primaryButtonStateLeft = OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.LTouch) ? "True" : "False";
    string secondaryButtonStateLeft = OVRInput.Get(OVRInput.Button.SecondaryIndexTrigger, OVRInput.Controller.LTouch) ? "True" : "False";
    string primaryTouchStateLeft = OVRInput.Get(OVRInput.Touch.PrimaryThumbRest, OVRInput.Controller.LTouch) ? "True" : "False";
    string secondaryTouchStateLeft = OVRInput.Get(OVRInput.Touch.SecondaryThumbRest, OVRInput.Controller.LTouch) ? "True" : "False";

    string primaryButtonStateRight = OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.RTouch) ? "True" : "False";
    string secondaryButtonStateRight = OVRInput.Get(OVRInput.Button.SecondaryIndexTrigger, OVRInput.Controller.RTouch) ? "True" : "False";
    string primaryTouchStateRight = OVRInput.Get(OVRInput.Touch.PrimaryThumbRest, OVRInput.Controller.RTouch) ? "True" : "False";
    string secondaryTouchStateRight = OVRInput.Get(OVRInput.Touch.SecondaryThumbRest, OVRInput.Controller.RTouch) ? "True" : "False";

    // Trigger and grip values for both controllers
    float triggerValueLeft = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.LTouch);
    float gripValueLeft = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, OVRInput.Controller.LTouch);
    float triggerValueRight = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.RTouch);
    float gripValueRight = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, OVRInput.Controller.RTouch);

    // Log data to CSV
    string csvLine = $"{univ_timestamp},{leftPosition},{leftRotation},{leftVelocity},{leftAngularVelocity},{leftAcceleration},{leftAngularAcceleration},{rightPosition},{rightRotation},{rightVelocity},{rightAngularVelocity},{rightAcceleration},{rightAngularAcceleration},{hmdPos},{hmdRot},{hmdVel},{hmdAngVel},{hmdAccel},{hmdAngAccel},{leftEyePos},{leftEyeRot},{leftEyeVel},{leftEyeAngVel},{leftEyeAccel},{leftEyeAngAccel},{rightEyePos},{rightEyeRot},{rightEyeVel},{rightEyeAngVel},{rightEyeAccel},{rightEyeAngAccel},{centerEyePos},{centerEyeRot},{centerEyeVel},{centerEyeAngVel},{centerEyeAccel},{centerEyeAngAccel},{primaryButtonStateLeft},{secondaryButtonStateLeft},{primaryTouchStateLeft},{secondaryTouchStateLeft},{primaryButtonStateRight},{secondaryButtonStateRight},{primaryTouchStateRight},{secondaryTouchStateRight},{triggerValueLeft},{gripValueLeft},{triggerValueRight},{gripValueRight}";
    csvData.Add(csvLine);

    // Update previous values for the next frame
    previousHmdVelocity = hmdVelocity;
    previousHmdAngularVelocity = hmdAngularVelocity;
    previousLeftEyeVelocity = leftEyeVelocity;
    previousLeftEyeAngularVelocity = leftEyeAngularVelocity;
    previousRightEyeVelocity = rightEyeVelocity;
    previousRightEyeAngularVelocity = rightEyeAngularVelocity;
    previousCenterEyeVelocity = centerEyeVelocity;
    previousCenterEyeAngularVelocity = centerEyeAngularVelocity;
    previousTime = currentTime;

    // Write a condition to create a file with the video name and push stuff in it
    // Create a time global time instance and then time stamps relevant to it

    if (write_data == true) // Time to write data as a scene ends
    {
        // Use a writing data function with video name

        name = trials[conditionCounter].video_id; // The current video name
        filePath = Application.dataPath + "/" + name + ".csv";

        using (TextWriter tw = new StreamWriter(filePath, false))
        {
            // Write the header
            tw.WriteLine("Timestamp,LeftPositionX,LeftPositionY,LeftPositionZ,LeftRotationX,LeftRotationY,LeftRotationZ,LeftRotationW,LeftVelocityX,LeftVelocityY,LeftVelocityZ,LeftAngularVelocityX,LeftAngularVelocityY,LeftAngularVelocityZ,LeftAccelerationX,LeftAccelerationY,LeftAccelerationZ,LeftAngularAccelerationX,LeftAngularAccelerationY,LeftAngularAccelerationZ,RightPositionX,RightPositionY,RightPositionZ,RightRotationX,RightRotationY,RightRotationZ,RightRotationW,RightVelocityX,RightVelocityY,RightVelocityZ,RightAngularVelocityX,RightAngularVelocityY,RightAngularVelocityZ,RightAccelerationX,RightAccelerationY,RightAccelerationZ,RightAngularAccelerationX,RightAngularAccelerationY,RightAngularAccelerationZ,HMDPositionX,HMDPositionY,HMDPositionZ,HMDRotationX,HMDRotationY,HMDRotationZ,HMDRotationW,HMDVelocityX,HMDVelocityY,HMDVelocityZ,HMDAngularVelocityX,HMDAngularVelocityY,HMDAngularVelocityZ,HMDAccelerationX,HMDAccelerationY,HMDAccelerationZ,HMDAngularAccelerationX,HMDAngularAccelerationY,HMDAngularAccelerationZ,LeftEyePositionX,LeftEyePositionY,LeftEyePositionZ,LeftEyeRotationX,LeftEyeRotationY,LeftEyeRotationZ,LeftEyeRotationW,LeftEyeVelocityX,LeftEyeVelocityY,LeftEyeVelocityZ,LeftEyeAngularVelocityX,LeftEyeAngularVelocityY,LeftEyeAngularVelocityZ,LeftEyeAccelerationX,LeftEyeAccelerationY,LeftEyeAccelerationZ,LeftEyeAngularAccelerationX,LeftEyeAngularAccelerationY,LeftEyeAngularAccelerationZ,RightEyePositionX,RightEyePositionY,RightEyePositionZ,RightEyeRotationX,RightEyeRotationY,RightEyeRotationZ,RightEyeRotationW,RightEyeVelocityX,RightEyeVelocityY,RightEyeVelocityZ,RightEyeAngularVelocityX,RightEyeAngularVelocityY,RightEyeAngularVelocityZ,RightEyeAccelerationX,RightEyeAccelerationY,RightEyeAccelerationZ,RightEyeAngularAccelerationX,RightEyeAngularAccelerationY,RightEyeAngularAccelerationZ,CenterEyePositionX,CenterEyePositionY,CenterEyePositionZ,CenterEyeRotationX,CenterEyeRotationY,CenterEyeRotationZ,CenterEyeRotationW,CenterEyeVelocityX,CenterEyeVelocityY,CenterEyeVelocityZ,CenterEyeAngularVelocityX,CenterEyeAngularVelocityY,CenterEyeAngularVelocityZ,CenterEyeAccelerationX,CenterEyeAccelerationY,CenterEyeAccelerationZ,CenterEyeAngularAccelerationX,CenterEyeAngularAccelerationY,CenterEyeAngularAccelerationZ,PrimaryButtonStateLeft,SecondaryButtonStateLeft,PrimaryTouchStateLeft,SecondaryTouchStateLeft,PrimaryButtonStateRight,SecondaryButtonStateRight,PrimaryTouchStateRight,SecondaryTouchStateRight,TriggerValueLeft,GripValueLeft,TriggerValueRight,GripValueRight");

            // Write the data
            foreach (string line in csvData)
            {
                tw.WriteLine(line);
            }
        }

        write_data = false;
        csvData.Clear(); // Empty the file
    }
}

}