using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PlayFab;
using PlayFab.ClientModels;
using PlayFab.DataModels;
using PlayFab.ProfilesModels;
using System;

public class PlayFabController : MonoBehaviour
{
    public static PlayFabController PFC;
    private int pressed = 0;
    private float startTime;
    string pressTIME;
    string releaseTIME;
    public List<string> ButtonDataList = new List<string>();
    ConditionController conditionController; 
    CarMovement carMovement;
    public GameObject warningCanvas; 

    public int counter; 
    public int warningCounter; 
    private void Start()
    {
        conditionController = GameObject.Find("ConditionController").GetComponent<ConditionController>();
        carMovement = GameObject.Find("CarMovement").GetComponent<CarMovement>();
    }
    public float deltaTime2;
    void FixedUpdate()
    {       
        float deltaTime = Time.time - carMovement.startTime;
        
        float distance = carMovement.carDistance; 
        double distanceDouble = System.Math.Round(distance, 2);
        if (conditionController.trial)
        {
            if (Input.GetMouseButton(0) && pressed == 0)        // PRESS
            {
                pressTIME = System.Math.Round(deltaTime, 2).ToString();
                ButtonDataList.Add("P: " + pressTIME + ": " + distanceDouble);
                pressed = 1;
            }
            if (!Input.GetMouseButton(0) && pressed == 1)       // RELEASE
            {
                deltaTime2 = Time.time;
                
                releaseTIME = System.Math.Round(deltaTime, 2).ToString();
                ButtonDataList.Add("R: " + releaseTIME + ": " + distanceDouble);
                SetUserData();
                pressed = 0;
            }

            if (Time.time - deltaTime2 > 15 && pressed == 0)
                {
                warningCounter += 1; 
                if (warningCounter == 1)
                {
                    StartCoroutine(Warning());       
                }                                                     
            }                         
        }

    }

    IEnumerator Warning()
    {
        // warningCanvas.SetActive(true);
        yield return new WaitForSecondsRealtime(1f);
        // warningCanvas.SetActive(false);
        deltaTime2 = Time.time;
        warningCounter = 0;

    }

    void SetUserData()
    {
        string DataString = String.Join("; " , ButtonDataList);
       

        if (conditionController.conditionCounter == 1)
        {
            PlayFabClientAPI.UpdateUserData(new UpdateUserDataRequest()
            {
                Data = new Dictionary<string, string>() {
            {"Condition 1: baseline", DataString},
            {"Participant ID", PlayerPrefs.GetString("ID")}
            }
            },
            result => Debug.Log("Successfully updated user data"),
            error =>
            {
                Debug.Log("Got error setting user data Ancestor to Arthur");
                Debug.Log(error.GenerateErrorReport());
            });
        }

        if (conditionController.conditionCounter == 2)
        {
            PlayFabClientAPI.UpdateUserData(new UpdateUserDataRequest()
            {
                Data = new Dictionary<string, string>() {
            {"Condition 2: bumper eHMI", DataString},

            }
            },
            result => Debug.Log("Successfully updated user data"),
            error =>
            {
                Debug.Log("Got error setting user data Ancestor to Arthur");
                Debug.Log(error.GenerateErrorReport());
            });
        }

        if (conditionController.conditionCounter == 3)
        {
            PlayFabClientAPI.UpdateUserData(new UpdateUserDataRequest()
            {
                Data = new Dictionary<string, string>() {
            {"Condition 3: tracker", DataString},

            }
            },
            result => Debug.Log("Successfully updated user data"),
            error =>
            {
                Debug.Log("Got error setting user data Ancestor to Arthur");
                Debug.Log(error.GenerateErrorReport());
            });
        }

        if (conditionController.conditionCounter == 4)
        {
            PlayFabClientAPI.UpdateUserData(new UpdateUserDataRequest()
            {
                Data = new Dictionary<string, string>() {
            {"Condition 4: progress", DataString},

            }
            },
            result => Debug.Log("Successfully updated user data"),
            error =>
            {
                Debug.Log("Got error setting user data Ancestor to Arthur");
                Debug.Log(error.GenerateErrorReport());
            });
        }

        if (conditionController.conditionCounter == 5)
        {
            PlayFabClientAPI.UpdateUserData(new UpdateUserDataRequest()
            {
                Data = new Dictionary<string, string>() {
            {"Condition 5: projection", DataString},

            }
            },
            result => Debug.Log("Successfully updated user data"),
            error =>
            {
                Debug.Log("Got error setting user data Ancestor to Arthur");
                Debug.Log(error.GenerateErrorReport());
            });
        }

    }
}
