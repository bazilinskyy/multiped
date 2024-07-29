using System.Collections;
using System.Collections.Generic;
using System.IO;
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

public class DemoCondition: MonoBehaviour
{

    public int Condition = 1;               // Set condition

    // Car movement
    public Transform myObject;
    public Spline FirstSpline; public Spline SecondSpline; public Spline ThirdSpline; public Spline FourthSpline; public Spline FullSpline;
    public AnimationCurve FirstCurve; public AnimationCurve SecondCurve; public AnimationCurve FullCurve;
    TweenBase FirstTween; TweenBase SecondTween; TweenBase ThirdTween; TweenBase FourthTween; TweenBase FullTween;
    private float firstAni = 9.3f; private float secAni = 5f; private float secDel = 12.8f; private int Ani = 12;
    private float wheelSize = 0.5f; private int firstDist = 115; private int secDist = 30; private int Dist = 145;
    public GameObject Lfront; public GameObject Lrear; public GameObject Rfront; public GameObject Rrear;
    // Wave variables
    private static int carCount = 0; private int Yield; public float startTime;
    public GameObject distance_cube;
    int[] array;
    int[] demoArray1 = { 1, 0, 2 };               // for single car demo's
    int[] trialArray = { 1, 2, 0, 2, 1, 2 };      // full trial array
    private bool yielding = false;
    public bool trial = false;
    public bool demo = false;
    public bool demo0 = false;
    public bool demo1 = false;
    public bool demo2 = false;
    public GameObject lightStripObject;
    public GameObject ReticlePointer;

    // UI first time
    public GameObject WelcomeCanvas;
    public GameObject WalkCanvas;
    public GameObject InfoCanvas;
    public GameObject InfoCanvas2;

    // UI
    public GameObject WalkCanvas2;
    public GameObject startDemoCanvas;
    public GameObject startTrialCanvas;
    public GameObject endTrialCanvas;
    public GameObject WillingnessToCross;

    private void Awake()
    {
        StartCoroutine(ActivatorVR("cardboard"));
        //StartCoroutine(ActivatorVR("none"));
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

    void Start()
    {
        lightStripObject.SetActive(false);
        WelcomeCanvas.SetActive(true);
    }

    public void PressMeButton()
    {
        WelcomeCanvas.SetActive(false);
        WalkCanvas.SetActive(true);
    }

    public void WalkButton()
    {
        WalkCanvas.SetActive(false);
        StartCoroutine(WaitAndWalk());
    }
    IEnumerator WaitAndWalk()
    {
        yield return new WaitForSecondsRealtime(0.2f);
        GameObject.Find("CameraHolder").GetComponent<MoveCamera>().StartWalk = true;
        yield return new WaitForSecondsRealtime(4.0f);
        InfoCanvas.SetActive(true);
    }

    public void Demo0Button1()
    {
        array = demoArray1;
        demo0 = true;
        StartCoroutine("Wave");
        InfoCanvas.SetActive(false);

    }

    public void Demo0Button2()
    {
        InfoCanvas2.SetActive(false);
        AsyncOperation asyncLoad = SceneManager.LoadSceneAsync("BaseEHMI");
    }


    // Demo & Trial
    public void WalkButton2()
    {
        WalkCanvas2.SetActive(false);
        StartCoroutine(WaitAndWalk());
    }

    public void StartDemoButton()
    {
        array = demoArray1;
        demo = true;
        startDemoCanvas.SetActive(false);
        StartCoroutine("Wave");
    }

    public void StartTrialButton()
    {
        array = trialArray;
        startTrialCanvas.SetActive(false);            // Disable canvas
        trial = true;                           // Set trial status to true
        StartCoroutine("Wave");
    }
    public void EndTrialButton()
    {
        endTrialCanvas.SetActive(false);
        AsyncOperation asyncLoad = SceneManager.LoadSceneAsync("BaseEHMI");
    }

    

    IEnumerator Wave()
    {
        for (; ; )
        {
            //If we haven't reached the maximum amount of cars yet
            if (carCount < array.Length)
            {
                ReticlePointer.SetActive(false);
                WillingnessToCross.SetActive(true);
                yielding = false;

                Yield = array[carCount]; //Check the array to see if the car should yield

                startTime = Time.time; // Start time of car route
                DriveCar();
                carCount += 1;

            }
            else
            {                // END 
                if (trial)
                {
                    trial = false;
                    endTrialCanvas.SetActive(true);
                }

                if (demo)
                {
                    demo = false;
                    startTrialCanvas.SetActive(true);
                }

                if (demo0)
                {
                    InfoCanvas2.SetActive(true);
                    demo0 = false;
                }

                ReticlePointer.SetActive(true);
                WillingnessToCross.SetActive(false);
                //LightStripBumper.Started = 0;
                StopCoroutine("Wave");
                carCount = 0;
            }

            //Delay until next vehicle starts
            if (Yield > 0)
            {
                yield return new WaitForSecondsRealtime(17.8f);
            }
            else
            {
                yield return new WaitForSecondsRealtime(12f);
            }
        }
    }

    public void DriveCar()
    {
        if (Yield > 0)
        {

            if (Yield == 1)
            {
                FirstTween = Tween.Spline(FirstSpline, myObject, 0, 1, true, firstAni, 0, FirstCurve, Tween.LoopType.None);
                SecondTween = Tween.Spline(SecondSpline, myObject, 0, 1, true, secAni, secDel, SecondCurve, Tween.LoopType.None);

                //wheels
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
                ThirdTween = Tween.Spline(ThirdSpline, myObject, 0, 1, true, firstAni, 0, FirstCurve, Tween.LoopType.None);
                FourthTween = Tween.Spline(FourthSpline, myObject, 0, 1, true, secAni, secDel, SecondCurve, Tween.LoopType.None);

                //wheels
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
            FullTween = Tween.Spline(FullSpline, myObject, 0, 1, true, Ani, 0, FullCurve, Tween.LoopType.None);
            WheelSpin0 LF = new WheelSpin0(Lfront, FullCurve, Dist, wheelSize); LF.SetupTween(Ani, 0);
            WheelSpin0 LR = new WheelSpin0(Lrear, FullCurve, Dist, wheelSize); LR.SetupTween(Ani, 0);
            WheelSpin0 RF = new WheelSpin0(Rfront, FullCurve, Dist, wheelSize); RF.SetupTween(Ani, 0);
            WheelSpin0 RR = new WheelSpin0(Rrear, FullCurve, Dist, wheelSize); RR.SetupTween(Ani, 0);
        }
    }
}
public class WheelSpin0
{
    public GameObject Wheel;
    public AnimationCurve myAnimation;
    public float Distance;
    public float WheelDiameter;
    public WheelSpin0(GameObject trans, AnimationCurve curve, float dist, float wheel)
    {
        Wheel = trans; myAnimation = curve; Distance = dist; WheelDiameter = wheel;
    }
    public void SetupTween(float duration, float delay)
    {
        float WheelDist = Mathf.PI * WheelDiameter;
        float Rotations = Distance / WheelDist;
        float deg = Rotations * 360;
        LeanTween.rotateAroundLocal(Wheel, Vector3.right, -deg, duration)
            .setEase(myAnimation)
            .setDelay(delay);
    }
}
