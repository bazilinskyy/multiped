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

public class Condition2 : MonoBehaviour
{
    public int Condition = 2;               // Set condition

    // Car movement
    public Transform myObject;
    public Spline FirstSpline; public Spline SecondSpline; public Spline ThirdSpline; public Spline FourthSpline; public Spline FullSpline;
    public AnimationCurve FirstCurve; public AnimationCurve SecondCurve; public AnimationCurve FullCurve;
    TweenBase FirstTween; TweenBase SecondTween; TweenBase ThirdTween; TweenBase FourthTween; TweenBase FullTween;
    private float firstAni = 9.3f; private float secAni = 5f; private float secDel = 12.8f; private int Ani = 12;
    private float wheelSize = 0.5f; private int firstDist = 115; private int secDist = 30; private int Dist = 145;
    public GameObject Lfront; public GameObject Lrear; public GameObject Rfront; public GameObject Rrear;
    // Wave variables
    private static int carCount = 0; private int Yield; private float startTime;
    public GameObject distance_cube;
    int[] array;
    int[] demoArray = { 1 };                      // for single car demo's
    int[] trialArray = { 1, 2, 0, 2, 1, 2 };      // full trial array
    private bool yielding = false;
    public bool trial = false;
    public bool demo = false;

    // LED Bumper
    public GameObject lightStripObject;
    LightStripBumper lightStripBumper;      // script
    public bool trialStart; 

    // VR pointer
    public GameObject ReticlePointer;

    public float pedestrian1_distance;      // For calculating moment to start animation pedestrian 1
    public float pedestrian2_distance;      // For calculating moment to start animation pedestrian 2
    public int counter = 0;
    public GameObject pedestrian1_position;
    public GameObject pedestrian2_position;

    // Projection light
    GameObject[] projections;
    public Vector3 endProjection;
    public GameObject zebra1;
    public GameObject zebra2;
    GameObject zebra;
    Transform projectionObject1;
    Transform projectionObject2;
    Transform projectionObject;

    // Tracker light
    public Vector3[] Positions;
    public Vector3[] Positions2;
    public Vector3[] Positions3;
    public Vector3[] EndPos;
    public GameObject tracker;
    Vector3 endProjection1;
    Vector3 endProjection2;
    Transform ProgressObject;
    Transform ProgressObject2;
    Transform TrackerObject;
    

    // UI

    void Start()
    {
        lightStripObject.SetActive(false);
        //lightStripBumper = GameObject.Find("LightStrip").GetComponent<LightStripBumper>();
        endProjection1 = Positions2[1];
        endProjection2 = Positions3[1];

    }

    public void WalkButton()
    {
        //WalkCanvas.SetActive(false);
        StartCoroutine(WaitAndWalk());
    }

    IEnumerator WaitAndWalk()
    {
        yield return new WaitForSecondsRealtime(0.2f);
        GameObject.Find("CameraHolder").GetComponent<MoveCamera>().StartWalk = true;
        yield return new WaitForSecondsRealtime(3.0f);
        //InfoCanvas.SetActive(true);
    }

    public void Demo()
    {
        array = demoArray;
        demo = true;
        //demoStart.SetActive(false);
        //trialStart.SetActive(false);

        StartCoroutine("Wave");
        trial = true;
    }

    public void StartTrial()
    {
        array = trialArray;
        //trialStart.SetActive(false);            // Disable canvas
        trial = true;                           // Set trial status to true

        ReticlePointer.SetActive(false);
        StartCoroutine("Wave");
    }
    public void TrialEnd()
    {
        //trialEnd.SetActive(false);
        //demoStart.SetActive(true);
    }













    IEnumerator ProjectionLightRunner()
    {
        float frametime = 0.066f;
        projectionObject1 = GameObject.Find("Projection").transform;
        projectionObject2 = GameObject.Find("Projection2").transform;

        if (Yield == 1)
        {
            zebra = zebra1;
            projectionObject = projectionObject1;
            projectionObject.transform.localPosition = Positions2[0];
            endProjection = endProjection1;
        }
        if (Yield == 2)
        {
            if (Condition == 5)
            {
                projectionObject = projectionObject2;
                zebra = zebra2;
                projectionObject.transform.localPosition = Positions3[0];
                endProjection = endProjection2;
            }
        }
        if (counter == 1)
        {
            foreach (Transform child in projectionObject)
            {
                if (child.gameObject.tag == "Projection")
                {
                    child.gameObject.SetActive(true);
                    yield return new WaitForSecondsRealtime(frametime);
                    child.gameObject.SetActive(false);
                }
            }
            Debug.Log(Time.time - startTime);
            yield return new WaitForSecondsRealtime(0.1f);
            zebra.SetActive(true);
            yield return new WaitForSecondsRealtime(3.4f);
            zebra.SetActive(false);
            yield return new WaitForSecondsRealtime(5.0f);
            yielding = true;
        }
    }

    IEnumerator ProgressLightRunner()
    {
        GameObject CarObject = GameObject.Find("Car").gameObject;
        ProgressObject = CarObject.transform.GetChild(0).GetChild(4).transform;
        ProgressObject2 = CarObject.transform.GetChild(0).GetChild(5).transform;
        float frametime = 0.10f;
        int counter2 = 0;
        if (counter == 1)
        {
            foreach (Transform child in ProgressObject)
            {
                counter2 += 1;
                GameObject frame1 = child.gameObject;
                frame1.SetActive(true);
                yield return new WaitForSecondsRealtime(frametime);
                if (counter2 == 51)
                {
                    yield return new WaitForSecondsRealtime(3.5f);
                }
                frame1.SetActive(false);
            }
            foreach (Transform child in ProgressObject2)
            {
                GameObject frame1 = child.gameObject;
                frame1.SetActive(true);
                yield return new WaitForSecondsRealtime(frametime);
                frame1.SetActive(false);
            }
            yield return new WaitForSecondsRealtime(5.0f);
            yielding = true;
        }
    }

    IEnumerator TrackerRunner()
    {
        GameObject CarObject = GameObject.Find("Car").gameObject;
        TrackerObject = CarObject.transform.GetChild(0).GetChild(6).transform;

        if (counter == 1)
        {
            yield return new WaitForSecondsRealtime(13.0f);
            Debug.Log("after 13");
            TrackerObject.localPosition = Positions[0];
            tracker.SetActive(false);
            yielding = true;
        }
    }


    // Update is called once per frame
    void FixedUpdate()
    {
        pedestrian1_distance = Vector3.Distance(distance_cube.transform.position, pedestrian1_position.transform.position);
        pedestrian2_distance = Vector3.Distance(distance_cube.transform.position, pedestrian2_position.transform.position);

        if (Yield == 1)
        {
            if (pedestrian1_distance <= 46)
            {
                if (Condition == 3 && !yielding)
                {
                    counter += 1;
                    tracker.SetActive(true);
                    StartCoroutine(TrackerRunner());
                    TrackerObject.localPosition = Vector3.MoveTowards(TrackerObject.localPosition, Positions[1], 0.11f * Time.fixedDeltaTime);
                }

                if (Condition == 4)
                {
                    counter += 1;
                    StartCoroutine(ProgressLightRunner());

                }
                if (Condition == 5)
                {
                    counter += 1;
                    StartCoroutine(ProjectionLightRunner());
                    projectionObject.localPosition = Vector3.MoveTowards(projectionObject.transform.localPosition, endProjection, 15f * Time.fixedDeltaTime);

                }

            }
        }

        if (Yield == 2)
        {

            if (pedestrian2_distance <= 46)
            {
                if (Condition == 3 && !yielding)
                {
                    counter += 1;
                    tracker.SetActive(true);
                    StartCoroutine(TrackerRunner());
                    TrackerObject.localPosition = Vector3.MoveTowards(TrackerObject.localPosition, Positions[1], 0.11f * Time.fixedDeltaTime);
                }
                if (Condition == 4)
                {
                    counter += 1;
                    StartCoroutine(ProgressLightRunner());
                }
                if (Condition == 5)
                {
                    counter += 1;
                    StartCoroutine(ProjectionLightRunner());
                    projectionObject.localPosition = Vector3.MoveTowards(projectionObject.transform.localPosition, endProjection, 15f * Time.fixedDeltaTime);
                }
            }
        }

        

    }

    IEnumerator Wave()
    {
        for (; ; )
        {
            //If we haven't reached the maximum amount of cars yet
            if (carCount < array.Length)
            {
                yielding = false;
      
                if (demo)
                {
                    Yield = 1;
                }

                if (trial)
                {
                    Yield = trialArray[carCount]; //Check the array to see if the car should yield

                }

                lightStripObject.SetActive(true);
                runHMI();
            
                startTime = Time.time; // Start time of car route
               
                counter = 0;
                DriveCar();
                carCount += 1;

            }
            else
            {                // END 
                if (trial)
                {
                    //trialEnd.SetActive(true);
                    trial = false;
                }

               
                if (demo)
                {
                    demo = false;
                }

               // LightStripBumper.Started = 0;
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

    // To run the light strip HMI animation
    void runHMI()
    {
        if (Yield > 0)
        {
            lightStripBumper.yieldHMI();
        }
        else
        {
            lightStripBumper.driveHMI();
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
                WheelSpin2 LF = new WheelSpin2(Lfront, FirstCurve, firstDist, wheelSize); LF.SetupTween(firstAni, 0);
                WheelSpin2 LR = new WheelSpin2(Lrear, FirstCurve, firstDist, wheelSize); LR.SetupTween(firstAni, 0);
                WheelSpin2 RF = new WheelSpin2(Rfront, FirstCurve, firstDist, wheelSize); RF.SetupTween(firstAni, 0);
                WheelSpin2 RR = new WheelSpin2(Rrear, FirstCurve, firstDist, wheelSize); RR.SetupTween(firstAni, 0);

                WheelSpin2 LF2 = new WheelSpin2(Lfront, FirstCurve, secDist, wheelSize); LF2.SetupTween(secAni, secDel);
                WheelSpin2 LR2 = new WheelSpin2(Lrear, FirstCurve, secDist, wheelSize); LR2.SetupTween(secAni, secDel);
                WheelSpin2 RF2 = new WheelSpin2(Rfront, FirstCurve, secDist, wheelSize); RF2.SetupTween(secAni, secDel);
                WheelSpin2 RR2 = new WheelSpin2(Rrear, FirstCurve, secDist, wheelSize); RR2.SetupTween(secAni, secDel);
            }
            if (Yield == 2)
            {
                ThirdTween = Tween.Spline(ThirdSpline, myObject, 0, 1, true, firstAni, 0, FirstCurve, Tween.LoopType.None);
                FourthTween = Tween.Spline(FourthSpline, myObject, 0, 1, true, secAni, secDel, SecondCurve, Tween.LoopType.None);

                //wheels
                WheelSpin2 LF = new WheelSpin2(Lfront, FirstCurve, firstDist, wheelSize); LF.SetupTween(firstAni, 0);
                WheelSpin2 LR = new WheelSpin2(Lrear, FirstCurve, firstDist, wheelSize); LR.SetupTween(firstAni, 0);
                WheelSpin2 RF = new WheelSpin2(Rfront, FirstCurve, firstDist, wheelSize); RF.SetupTween(firstAni, 0);
                WheelSpin2 RR = new WheelSpin2(Rrear, FirstCurve, firstDist, wheelSize); RR.SetupTween(firstAni, 0);

                WheelSpin2 LF2 = new WheelSpin2(Lfront, FirstCurve, secDist, wheelSize); LF2.SetupTween(secAni, secDel);
                WheelSpin2 LR2 = new WheelSpin2(Lrear, FirstCurve, secDist, wheelSize); LR2.SetupTween(secAni, secDel);
                WheelSpin2 RF2 = new WheelSpin2(Rfront, FirstCurve, secDist, wheelSize); RF2.SetupTween(secAni, secDel);
                WheelSpin2 RR2 = new WheelSpin2(Rrear, FirstCurve, secDist, wheelSize); RR2.SetupTween(secAni, secDel);
            }
        }
        if (Yield == 0)
        {
            FullTween = Tween.Spline(FullSpline, myObject, 0, 1, true, Ani, 0, FullCurve, Tween.LoopType.None);
            WheelSpin2 LF = new WheelSpin2(Lfront, FullCurve, Dist, wheelSize);
            LF.SetupTween(Ani, 0);
            WheelSpin2 LR = new WheelSpin2(Lrear, FullCurve, Dist, wheelSize);
            LR.SetupTween(Ani, 0);
            WheelSpin2 RF = new WheelSpin2(Rfront, FullCurve, Dist, wheelSize);
            RF.SetupTween(Ani, 0);
            WheelSpin2 RR = new WheelSpin2(Rrear, FullCurve, Dist, wheelSize);
            RR.SetupTween(Ani, 0);
        }
    }
}

public class WheelSpin2
{
    public GameObject Wheel;
    public AnimationCurve myAnimation;
    public float Distance;
    public float WheelDiameter;
    public WheelSpin2(GameObject trans, AnimationCurve curve, float dist, float wheel)
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
