using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using System;
using UnityEngine.XR;

public class ConditionController : MonoBehaviour
{
    // Condition arrays
    int[] order1 =  {0, 1, 2, 3, 4, 5 };           
    int[] order2 =  {0, 1, 2, 3, 5, 4 };
    int[] order3 =  {0, 1, 2, 4, 3, 5 };
    int[] order4 =  {0, 1, 2, 4, 5, 3 };
    int[] order5 =  {0, 1, 2, 5, 3, 4 };
    int[] order6 =  {0, 1, 2, 5, 4, 3 };

    int[] order7 =  {0, 1, 3, 2, 4, 5 };
    int[] order8 =  {0, 1, 3, 2, 5, 4 };
    int[] order9 =  {0, 1, 3, 4, 2, 5 };
    int[] order10 = {0, 1, 3, 4, 5, 2 };
    int[] order11 = {0, 1, 3, 5, 2, 4 };
    int[] order12 = {0, 1, 3, 5, 4, 2 };

    int[] order13 = {0, 1, 4, 2, 3, 5 };
    int[] order14 = {0, 1, 4, 2, 5, 3 };
    int[] order15 = {0, 1, 4, 3, 2, 5 };
    int[] order16 = {0, 1, 4, 3, 5, 2 };
    int[] order17 = {0, 1, 4, 5, 2, 3 };
    int[] order18 = {0, 1, 4, 5, 3, 2 };

    int[] order19 = {0, 1, 5, 2, 3, 4 };
    int[] order20 = {0, 1, 5, 2, 4, 3 };
    int[] order21 = {0, 1, 5, 3, 2, 4 };
    int[] order22 = {0, 1, 5, 3, 4, 2 };
    int[] order23 = {0, 1, 5, 4, 2, 3 };
    int[] order24 = {0, 1, 5, 4, 3, 2 };

    int[] orderArray; 
    string order; 
    public int conditionBlock = 0; 
    public bool conditionFinished = false;
    LightStripBumper lightStripBumper;      // script
    public GameObject LEDBumperObject;
    public GameObject tracker;
    public GameObject progress; 
    public GameObject projection; 
    
    CarMovement carMovementScript;
    PlayFabController playfabScript;
    public int conditionCounter = 0;

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

    public GameObject WillingnessToCross;
    public GameObject reticle;
    public GameObject CountDown; 
    public bool preview = false; 
    public bool trial = false;

    public AudioSource buttonSound;

    public void Start()
    {           
        StartCoroutine(ActivatorVR("cardboard"));
        buttonSound = GetComponent<AudioSource>();
        Start2();       
    }
    public IEnumerator ActivatorVR(string YESVR)
    {
        
        XRSettings.LoadDeviceByName(YESVR);
        yield return null;
        XRSettings.enabled = true;
    }
    public IEnumerator DectivatorVR(string NOVR)
    {
        XRSettings.LoadDeviceByName(NOVR);
        yield return null;
        XRSettings.enabled = false;       
    }

    void Start2()
    {        
        order = PlayerPrefs.GetString("Password");
        
        if (order == "sdc1")             {
            orderArray = order1;    }
        if (order == "sdc2")             {
            orderArray = order2;    }
        if (order == "sdc3")             {
            orderArray = order3;    }
        if (order == "sdc4")             {
            orderArray = order4;    }
        if (order == "sdc5")             {
            orderArray = order5;    }
        if (order == "sdc6")             {
            orderArray = order6;    }
        if (order == "sdc7")             {
            orderArray = order7;    }
        if (order == "sdc8")             {
            orderArray = order8;    }
        if (order == "sdc9")             {
            orderArray = order9;    }
        if (order == "sdc10")             {
            orderArray = order10;    }
        if (order == "sdc11")             {
            orderArray = order11;    }
        if (order == "sdc12")             {
            orderArray = order12;    }
        if (order == "sdc13")             {
            orderArray = order13;    }
        if (order == "sdc14")             {
            orderArray = order14;    }
        if (order == "sdc15")             {
            orderArray = order15;    }
        if (order == "sdc16")             {
            orderArray = order16;    }
        if (order == "sdc17")             {
            orderArray = order17;    }
        if (order == "sdc18")             {
            orderArray = order18;    }
        if (order == "sdc19")             {
            orderArray = order19;    }
        if (order == "sdc20")             {
            orderArray = order20;    }
        if (order == "sdc21")             {
            orderArray = order21;    }
        if (order == "sdc22")             {
            orderArray = order22;    }
        if (order == "sdc23")             {
            orderArray = order23;    }
        if (order == "sdc24")             {
            orderArray = order24;    }

        conditionCounter = PlayerPrefs.GetInt("ConditionCount");      
        if (conditionCounter == 0)
        {
            orderArray = order1; 
        }     

        conditionBlock = orderArray[conditionCounter];

        carMovementScript = GameObject.Find("CarMovement").GetComponent<CarMovement>();
        playfabScript = GameObject.Find("PlayFabController").GetComponent<PlayFabController>();
        LEDBumperObject.SetActive(true);            // Turn on LED bumper
        tracker.SetActive(false);                   // Switch off tracker   
        progress.SetActive(false);                  // Switch off progressbar
        projection.SetActive(false);                // Switch off projection

        if (conditionBlock == 0)
        {
            LEDBumperObject.SetActive(false);
            DemoStart();
        }
        if (conditionBlock == 1)
        {
            demoTitle.text = "Block " + (conditionCounter.ToString());            
            demoText.text = "In this block you will encounter self-driving cars.";
            LEDBumperObject.SetActive(false);
            TrialStart();
        }
        if (conditionBlock == 2)
        {
            demoTitle.text = "Block " + (conditionCounter.ToString());            
            demoText.text = "In this block you will encounter self-driving cars with a light on the bumper. This light will be fully lit when driving, " +
                "but will show a pulsating animation when intending to stop. ";
            TrialStart();
        }
        if (conditionBlock == 3)
        {
            demoTitle.text = "Block " + (conditionCounter.ToString());           
            demoText.text = "In this block you will encounter self-driving cars with a light on the bumper, plus a windshield display. While braking, this display points into the direction of the " +
                "pedestrian the car is stopping for. ";
            TrialStart();
            tracker.SetActive(true);
        }
        if (conditionBlock == 4)
        {
            demoTitle.text = "Block " + (conditionCounter.ToString());           
            demoText.text = "In this block you will encounter self-driving cars with a light on the bumper, plus a progress display. This progress display lights up on the front window while braking, " +
                "and the closer the car gets to its stopping location, the more the display increases in size. ";
            progress.SetActive(true);
            TrialStart();
        }
        if (conditionBlock == 5)
        {
            demoTitle.text = "Block " + (conditionCounter.ToString());
            demoText.text = "In this block you will encounter self-driving cars with a light on the bumper, plus a projection display. This projection display projects arrows on the road, which point to the " +
                   "stopping location of the car. When the car comes to a stop it will show a pedestrian crossing. ";
            projection.SetActive(true);
            TrialStart();
        }

        trialTitle.text = demoTitle.text;

    }
    private void FixedUpdate()
    {
        if (carMovementScript.conditionFinished)
        {
            if (conditionBlock == 0)
            {
                // enable last canvas
                WillingnessToCross.SetActive(false);
                reticle.SetActive(true);
                demoInfoCanvas2.SetActive(true);
                carMovementScript.conditionFinished = false;             
            }
            if (conditionBlock > 0)
            {
                if (preview)
                {
                    WillingnessToCross.SetActive(false);
                    reticle.SetActive(true);
                    trialStartCanvas.SetActive(true);
                    carMovementScript.conditionFinished = false;
                    preview = false;
                }
                if (!preview && trial)
                {
                    if (conditionCounter < 5)
                    {
                        WillingnessToCross.SetActive(false);
                        reticle.SetActive(true);
                        trialEndCanvas.SetActive(true);
                        carMovementScript.conditionFinished = false;
                        trial = false;
                    }
                    else
                    {
                        WillingnessToCross.SetActive(false);
                        carMovementScript.conditionFinished = false;
                        trial = false;
                        ExperimentEndCanvas.SetActive(true);
                    }
                }
            }
        }
    }

    // UI DEMO
    void DemoStart()
    {
        demoWelcomeCanvas.SetActive(true);
    }
    public void DemoCanvas1()
    {
        demoWelcomeCanvas.SetActive(false);
        demoWalkCanvas.SetActive(true);
    }
    public void DemoCanvas2()
    {
        demoWalkCanvas.SetActive(false);
        StartCoroutine(WalkForward());
        demoInfoCanvas1.SetActive(true);
    }
    public void DemoCanvas3()
    {
        demoInfoCanvas1.SetActive(false);
        StartCoroutine(CountDownDemo());      
    }
    public void DemoCanvas4()
    {
        demoInfoCanvas2.SetActive(false);      
        StartCoroutine(ActivatorVR("none"));
        SceneManager.LoadScene("Environment");
    }

    IEnumerator CountDownDemo()
    {
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
        yield return new WaitForSecondsRealtime(0.2f);
        GameObject.Find("CameraHolder").GetComponent<MoveCamera>().StartWalk = true;
        yield return new WaitForSecondsRealtime(3.0f);
    }

    // UI TRIALS
    void TrialStart()
    {
        trialWalkCanvas.SetActive(true);
    }
    public void TrialCanvas1()
    {
        trialWalkCanvas.SetActive(false);
        StartCoroutine(WalkForward());
        trialDemoCanvas.SetActive(true);
    }
    public void TrialCanvas2()                  // Start preview
    {
        trialDemoCanvas.SetActive(false);
        preview = true;
        StartCoroutine(CountDownPreview());
    }

    IEnumerator CountDownPreview()
    {
        reticle.SetActive(false);
        CountDown.SetActive(true);
        carMovementScript.CountSound.Play();
        yield return new WaitForSecondsRealtime(3f);
        carMovementScript.AudioBeep.Play();
        yield return new WaitForSecondsRealtime(1f);
        carMovementScript.StartCarPreview();
        WillingnessToCross.SetActive(true);
    }
    public void TrialCanvas3()                  // Start trial
    {
        trialStartCanvas.SetActive(false);
        StartCoroutine(CountDownTrial());           
    }

    IEnumerator CountDownTrial()
    {
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
            // Set next condition        
            PlayerPrefs.SetInt("ConditionCount", conditionCounter + 1);
            trialEndCanvas.SetActive(false);
            StartCoroutine(ActivatorVR("none"));
            SceneManager.LoadScene("Environment");        
    }

    public void RestartPreview()
    {
        trialStartCanvas.SetActive(false);
        trialDemoCanvas.SetActive(true);
    }

    public void Reset0()
    {
        PlayerPrefs.SetInt("ConditionCount", 1);
        StartCoroutine(ActivatorVR("none"));
        SceneManager.LoadScene("Environment");
    }

    public void ButtonSound()
    {
        buttonSound.Play(); 
    }
}
