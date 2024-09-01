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

public class CarMovement : MonoBehaviour
{
    // Car movement
    public Transform myObject;
    public Spline FirstSpline; public Spline SecondSpline; public Spline ThirdSpline; public Spline FourthSpline; public Spline FullSpline;
    public AnimationCurve FirstCurve; public AnimationCurve SecondCurve; public AnimationCurve FullCurve;
    TweenBase FirstTween; TweenBase SecondTween; TweenBase ThirdTween; TweenBase FourthTween; TweenBase FullTween;

    private float firstAni = 11f;         // For yielding to pedestrian 1
    private float secDel = 14f;           // For yielding to pedestrian 1
    private int firstDist = 115;          // For yielding to pedestrian 1

    public GameObject measuringPoint;
    public float carDistance; 

    private float secAni = 5f; private int Ani = 12;
    private float wheelSize = 0.5f; private int secDist = 30; private int Dist = 145;
    public GameObject Lfront; public GameObject Lrear; public GameObject Rfront; public GameObject Rrear;
    public int carCount = 0;
    public int Yield;          // For yielding to pedestrian 1, 2, or not at all (0)
    public float startTime;     // Start time of vehicles
    
    int[] yieldArrayDemo = {1, 2, 0};
    int[] yieldArrayPreview = {1};

    int[] yieldArray;
    int[] yieldArrayCondition1 = { 0 };
    int[] yieldArrayCondition2 = { 1 };
    int[] yieldArrayCondition3 = { 2 };
    int[] yieldArrayCondition4 = { 0 };
    int[] yieldArrayCondition5 = { 0 };

    public bool WaveStarted = false;

    public GameObject distance_cube;
    public GameObject pedestrian1;
    public GameObject pedestrian2;
    public float pedestrian1_distance;      // For calculating moment to start animation pedestrian 1
    public float pedestrian2_distance;      // For calculating moment to start animation pedestrian 2
    public float pedestrian1_distance_x;
    public float pedestrian2_distance_x;
    public float speed;
    int counter;
    public bool yielding; 
    float fixedDeltaTime;
    LightStripBumper LEDscript;
    public bool conditionFinished = false;
    ConditionController conditionScript; 
    PlayFabController playfabScript;

    public AudioSource AudioBeep; 
    public AudioSource CountSound; 


    public void Awake()
    {
        AudioBeep = GetComponent<AudioSource>();
    }

    public void StartCarDemo()
    {
        carCount = 0;
        yieldArray = yieldArrayDemo;
        conditionScript = GameObject.Find("ConditionController").GetComponent<ConditionController>();
        StartCoroutine("Wave");
        StartCoroutine(SpeedCalculator());
    }
    public void StartCarPreview()
    {
        
        //InvokeRepeating("OutputTime", 0, 0.5f);  //0.5s delay, repeat every 0.5s              
        carCount = 0;
        yieldArray = yieldArrayPreview;
        conditionScript = GameObject.Find("ConditionController").GetComponent<ConditionController>();
        StartCoroutine("Wave");
        if (conditionScript.eHMIOn == 1)
        {
            LEDscript = GameObject.Find("LightStrip").GetComponent<LightStripBumper>();
        }
    }
    public void StartCar()
    {       
        carCount = 0;
        conditionScript = GameObject.Find("ConditionController").GetComponent<ConditionController>();
        playfabScript = GameObject.Find("PlayFabController").GetComponent<PlayFabController>();

        string time = System.DateTime.UtcNow.AddHours(2f).ToString();
        playfabScript.ButtonDataList.Add(time);       // Add time to playfab data
                      
        yieldArray = yieldArrayCondition1;

        // Trigger eHMI
        if (conditionScript.eHMIOn == 1)
        {
            Debug.Log("eHMI enabled");
            LEDscript = GameObject.Find("LightStrip").GetComponent<LightStripBumper>();
        } else {
            Debug.Log("eHMI disabled");
            LEDscript = GameObject.Find("LightStrip").GetComponent<LightStripBumper>();
            GameObject.Find("LightStrip").SetActive(false);
        }

        // Trigger yielding
        if (conditionScript.yielding == 1)
        {
            Debug.Log("Yielding ON for P1");
            Yield = 1;
        } else if (conditionScript.yielding == 2) {
            Debug.Log("Yielding ON for P2");
            Yield = 2;
        } else {
            Debug.Log("Yielding OFF");
            Yield = 0;
        }
        StartCoroutine("Wave");
    }

    public void FixedUpdate()
    {      
        fixedDeltaTime = Time.time - startTime;
        pedestrian1_distance = Vector3.Distance(distance_cube.transform.position, conditionScript.p1_object.transform.position);
        pedestrian2_distance = Vector3.Distance(distance_cube.transform.position, conditionScript.p2_object.transform.position);
        pedestrian1_distance_x = Mathf.Abs(distance_cube.transform.position.x - conditionScript.p1_object.transform.position.x);
        pedestrian2_distance_x = Mathf.Abs(distance_cube.transform.position.x - conditionScript.p2_object.transform.position.x);
        carDistance = Vector3.Distance(measuringPoint.transform.position, distance_cube.transform.position);

        StartCoroutine(SpeedCalculator()); 

        Debug.Log("continous_pedestrian1_distance= " + pedestrian1_distance_x + " pedestrian2_distance=" + pedestrian2_distance_x);
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
                //Debug.Log("Start yielding at: " + fixedDeltaTime + "; Distance: " + carDistance);
            }

            if (pedestrian1_distance < 3)
            {
                yielding = false;
            }
        }    
    }

    IEnumerator Wave()
    {
        for (; ; )
        {

            WaveStarted = true;         // For LED bumper
            //If we haven't reached the maximum amount of cars yet
            if (carCount < yieldArray.Length)
            {            
                // if (conditionScript.conditionCounter == 0)
                // {
                //     Yield = Random.Range(0, 3);
                //     //Debug.Log("Yield = " + Yield);
                // }
                // else
                // {
                //     Yield = yieldArray[carCount]; //Check the array to see if the car should yield
                // }           

                startTime = Time.time; // Start time of car route
                
                if (conditionScript.trial)
                {
                    playfabScript.ButtonDataList.Add("(" + (Yield).ToString() + ")");       // Add Yield number to playfab data
                }              
                DriveCar();                    
                carCount += 1;               
            }
            else
            {                // END 
                Debug.Log("car movement finished");
                conditionFinished = true;                
                StopCoroutine("Wave");               
            }

            //Delay until next vehicle starts
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
            if (conditionScript.conditionCounter > 1)
            {
                LEDscript.counter = 0;                  // Reset light strip
                LEDscript.counter2 = 0;                 //
            }
                      
            counter = 0;
        }
    }
    
    public void DriveCar()
    {
        if (Yield > 0)
        {
            if (Yield == 1)
            {
                firstAni = 11f;         // For yielding to pedestrian 1
                secDel = 14f;           // For yielding to pedestrian 1
                firstDist = 115;
                secAni = 5f;
                secDist = 30;

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
                    firstAni = 12f;         // For yielding to pedestrian 2
                    secDel = 15f;           // For yielding to pedestrian 2
                    firstDist = 125;        // For yielding to pedestrian 2
                
                    secAni = 4f;
                    secDist = 20;

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

    IEnumerator SpeedCalculator()
    {
        if (WaveStarted)
        {          
            Vector3 position1 = distance_cube.transform.position;
            yield return new WaitForSeconds(0.5f);
            Vector3 position2 = distance_cube.transform.position;
            float distance = Vector3.Distance(position1, position2);
            speed = distance * 2 * 3.6f;        
        }
    }

    void OutputTime()
    {
        Debug.Log(carDistance + "m; " + fixedDeltaTime + "s");
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
}

