using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using System;
using UnityEngine.XR;

public class ConditionController : MonoBehaviour
{
    public bool conditionFinished = false;
    LightStripBumper lightStripBumper;      // script
    public GameObject LEDBumperObject;
    public GameObject tracker;
    public GameObject progress; 
    public GameObject projection; 
    
    CarMovement carMovementScript;
    PlayFabController playfabScript;
    public int conditionCounter = 0;
    // todo: import conditions from csv
    public int numberConditions = 10; // todo: number of conditions

    public int eHMIOn = 0;   // 0=no 1=slowly-pulsing light band
    public int yielding = 0;     // 0=yes for P1 1=yes for P2 2=no
    public int distPed = 0;      // distance between P1 and P2 distances [2 .. +2 ..  20].

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
        Debug.Log("conditionCounter=" + conditionCounter);
        carMovementScript = GameObject.Find("CarMovement").GetComponent<CarMovement>();
        playfabScript = GameObject.Find("PlayFabController").GetComponent<PlayFabController>();
        LEDBumperObject.SetActive(true);            // Turn on LED bumper
        tracker.SetActive(false);                   // Switch off tracker   
        progress.SetActive(false);                  // Switch off progressbar
        projection.SetActive(false);                // Switch off projection

        if (conditionCounter == 0)
        {
            eHMIOn = 0;
            yielding = 0;
        }
        if (conditionCounter == 1)
        {
            eHMIOn = 0;
        }
        if (conditionCounter == 2)
        {
            eHMIOn = 1;
        }
        if (conditionCounter == 3)
        {
            // tracker.SetActive(true);
            eHMIOn = 1;
        }
        if (conditionCounter == 4)
        {
            // progress.SetActive(true);
            eHMIOn = 1;
        }
        if (conditionCounter == 5)
        {
            // projection.SetActive(true);
            eHMIOn = 1;
        }
        TrialStart();
        // trialTitle.text = demoTitle.text;

    }
    private void FixedUpdate()
    {
        // Debug.Log("carMovementScript.conditionFinished=" + carMovementScript.conditionFinished + " conditionBlock=" + conditionBlock + " conditionCounter=" + conditionCounter + " carMovementScript.conditionFinished=" + carMovementScript.conditionFinished + " trial=" + trial );
        if (carMovementScript != null) {
            if (carMovementScript.conditionFinished)
            {
                if (conditionCounter == numberConditions) {
                    ExperimentEndCanvas.SetActive(true);
                }
                Debug.Log("FixedUpdate::trial end");
                WillingnessToCross.SetActive(false);
                reticle.SetActive(true);
                // trialEndCanvas.SetActive(true);
                carMovementScript.conditionFinished = false;
                trial = false;
                conditionCounter = conditionCounter + 1;
                trialEndCanvas.SetActive(false);
                StartCoroutine(ActivatorVR("none"));
                // SceneManaxger.LoadScene("Environment");
                Start2();
                // TrialCanvas4();
                // if (conditionBlock == 0)
                // {
                //     Debug.Log("FixedUpdate::demo end");
                //     // enable last canvas
                //     WillingnessToCross.SetActive(false);
                //     reticle.SetActive(true);
                //     demoInfoCanvas2.SetActive(true);
                //     carMovementScript.conditionFinished = false;             
                // }
                // if (conditionBlock > 0)
                // {
                //     if (preview)
                //     {
                //         Debug.Log("FixedUpdate::preview end");
                //         WillingnessToCross.SetActive(false);
                //         reticle.SetActive(true);
                //         trialStartCanvas.SetActive(true);
                //         carMovementScript.conditionFinished = false;
                //         preview = false;
                //     }
                //     if (trial)
                //     {
                //         if (conditionCounter < 5)
                //         {
                //             Debug.Log("FixedUpdate::trial end");
                //             WillingnessToCross.SetActive(false);
                //             reticle.SetActive(true);
                //             trialEndCanvas.SetActive(true);
                //             carMovementScript.conditionFinished = false;
                //             trial = false;
                //             // TrialCanvas4();
                //         }
                //         else
                //         {
                //             Debug.Log("FixedUpdate::experiment end");
                //             WillingnessToCross.SetActive(false);
                //             carMovementScript.conditionFinished = false;
                //             trial = false;
                //             // ExperimentEndCanvas.SetActive(true);
                //         }
                //     }
                // }
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
    void TrialStart()
    {
        Debug.Log("TrialStart");
        TrialCanvas3();
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
    public void TrialCanvas3()                  // Start trial
    {
        Debug.Log("TrialCanvas3");
        // trialStartCanvas.SetActive(false);
        carMovementScript.AudioBeep.Play();
        trial = true;
        carMovementScript.StartCar();
        // StartCoroutine(CountDownTrial());
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
}
