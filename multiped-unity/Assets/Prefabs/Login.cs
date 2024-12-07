using System;
using System.Collections;
using System.Collections.Generic;
using PlayFab;
using PlayFab.ClientModels;
using UnityEngine;
using UnityEngine.SceneManagement; 
using UnityEngine.UI;

public class Login : MonoBehaviour
{
    public static string deviceModel;
    public static string operatingSystem;
    public static Resolution resolution;

    public string userName;
    public string playfabID;
    public string passWord;
    public InputField IDtextInput;
    public InputField IDtextInput2;

    public GameObject startPanel;
    public GameObject passwordPanel;
    public GameObject loginPanel;
    public GameObject infoPanel1;
    public GameObject infoPanel2;
    public GameObject infoPanel3;

    public GameObject SaveButton;
    public GameObject ContinueButton;
    public GameObject TypeText;
    public GameObject IDText;
    public GameObject InputField;
    public GameObject InputField2;

    public GameObject loading;
    public bool demo = false;

    public InputField password;
    public GameObject passCorrect;
    public GameObject passWrong;
    bool userNameAlreadySaved; 

    // *********************************** Connecting to PlayFab ***********************************
    public void Start()
    {
        //Note: Setting title Id here can be skipped if you have set the value in Editor Extensions already.
        if (string.IsNullOrEmpty(PlayFabSettings.TitleId))
        {
            PlayFabSettings.TitleId = "FA658"; // Please change this value to your own titleId from PlayFab Game Manager
        }

#if UNITY_ANDROID
        var requestAndroid = new LoginWithAndroidDeviceIDRequest { AndroidDeviceId = ReturnMobileID(), CreateAccount = true };
        PlayFabClientAPI.LoginWithAndroidDeviceID(requestAndroid, OnLoginSuccess, OnLoginFailure);
#endif
#if UNITY_IOS
        var requestIOS = new LoginWithIOSDeviceIDRequest { IOSDeviceId = ReturnMobileID(), CreateAccount = true};
        PlayFabClientAPI.LoginWithIOSDeviceID(requestIOS, OnLoginSuccess, OnLoginFailure);
#endif
    }
    public static string ReturnMobileID()
    {
        string deviceID = SystemInfo.deviceUniqueIdentifier;
        return deviceID;
    }
    private void OnLoginSuccess(LoginResult result)
    {
        Debug.Log("Succesfull login with DeviceID");
    }
    private void OnLoginFailure(PlayFabError error)
    {
        Debug.LogWarning("Something went wrong with your first API call.  :(");
        Debug.LogError("Here's some debug information:");
        Debug.LogError(error.GenerateErrorReport());
    }
    private void OnAddUsernameSuccess(AddUsernamePasswordResult result)
    {
        Debug.Log("Username saved");
    }
    private void OnAddUsernameFailure(PlayFabError error)
    {
        Debug.Log("Username already saved");
        Debug.LogError(error.GenerateErrorReport());
    }
     
    public void SetLoginSettings()
    {
        if (!userNameAlreadySaved)
        {         
          IDtextInput.text = userName;
          PlayerPrefs.SetString("ID", userName);
        }
        if (userNameAlreadySaved)
        {            
            TypeText.SetActive(false);
            InputField.SetActive(false);
            SaveButton.SetActive(false);
            ContinueButton.SetActive(true);
            IDText.SetActive(true);
            InputField2.SetActive(true);
            IDtextInput2.text = userName;
        }
    }
    public void SetUserName(string text)
    {
        userName = text;
        PlayerPrefs.SetString("ID", userName);
        Debug.Log(userName); 
    }

    public void Awake()
    {       
        startPanel.SetActive(true);
    }

    public void LoadPasswordPanel()
    {
        passWrong.SetActive(false);
        passCorrect.SetActive(false); 
        startPanel.SetActive(false);
        passwordPanel.SetActive(true); 
    }

    public void CheckPassword()
    {
        if (password.text.Contains("sdc"))
        {
            passWord = password.text;
            passWrong.SetActive(false);
            passCorrect.SetActive(true);
            StartCoroutine(WaitLogin());
        }
        else
        {
            passWrong.SetActive(true);
            StartCoroutine(Wait());
        }
    }
    public IEnumerator Wait()
    {
        yield return new WaitForSeconds(1.5f);
        passWrong.SetActive(false); 
    }

    public IEnumerator WaitLogin()
    {
        yield return new WaitForSeconds(1f);
        passCorrect.SetActive(false);
        PlayerPrefs.SetString("Password",password.text);
        LoadLogin();        
    }

    public void Back2Start()
    {
        startPanel.SetActive(true);
        passwordPanel.SetActive(false);
    }
    public void LoadLogin()
    {
        passwordPanel.SetActive(false);
        GetAccountInfo();
        loginPanel.SetActive(true);
    }
    public void AddUserName()
    {
        if (!userNameAlreadySaved)
        {
            var addUsername = new AddUsernamePasswordRequest { Email = ReturnMobileID() + "@gmail.com", Password = "123456", Username = userName};
            PlayFabClientAPI.AddUsernamePassword(addUsername, OnAddUsernameSuccess, OnAddUsernameFailure);
        }
        PlayerPrefs.SetInt("Demo", 0);
        loginPanel.SetActive(false);
        infoPanel1.SetActive(true);
    }

    public void LoadInfoPanel1()
    {
        // Demo
        PlayerPrefs.SetInt("Demo", 1);
        startPanel.SetActive(false); 
        infoPanel1.SetActive(true);
    }

    public void LoadInfoPanel2()
    {
        infoPanel1.SetActive(false);
        infoPanel2.SetActive(true);
    }

    public void LoadInfoPanel3()
    {
        infoPanel2.SetActive(false);
        infoPanel3.SetActive(true);
    }

    public void StartScene()
    {
        loading.SetActive(true);
        StartCoroutine(SceneStarter());
    }

    private IEnumerator SceneStarter()
    {
        Screen.orientation = ScreenOrientation.LandscapeLeft;
        if (PlayerPrefs.GetInt("Demo") == 1)
        {
            PlayerPrefs.SetInt("ConditionCount", 0);
        }
        if (PlayerPrefs.GetInt("Demo") == 0)
        {
            if (PlayerPrefs.GetInt("ConditionCount") == 0)
            {
                PlayerPrefs.SetInt("ConditionCount", 1);
            }                    
        }       
        AsyncOperation asyncLoad = SceneManager.LoadSceneAsync("Environment");
        
        // Wait until the asynchronous scene fully loads
        while (!asyncLoad.isDone)
        {
            yield return null;
        }
    }

    void GetAccountInfo()
    {
        GetAccountInfoRequest request = new GetAccountInfoRequest();
        PlayFabClientAPI.GetAccountInfo(request, Success, fail);
        SaveAndroidInfo(); 
    }

    void Success(GetAccountInfoResult result)
    {
        userName = result.AccountInfo.Username;                 // Username is saved to PlayFab player data.
        if (String.IsNullOrEmpty(userName))
        {
            userNameAlreadySaved = false; 
        }
        if (!String.IsNullOrEmpty(userName))
        {
            userNameAlreadySaved = true; 
        }
        SetLoginSettings(); 
    }

    void fail(PlayFabError error)
    {
        Debug.LogError(error.GenerateErrorReport());
    }

    void SaveAndroidInfo()
    {
        deviceModel = SystemInfo.deviceModel;
        operatingSystem = SystemInfo.operatingSystem;
        resolution = Screen.currentResolution;
        string resolutionString = resolution.ToString();
        PlayFabClientAPI.UpdateUserData(new UpdateUserDataRequest()
        {
            Data = new Dictionary<string, string>() {
            {"Password", passWord},
            {"DeviceModel", deviceModel},
            {"Operating system", operatingSystem},
            {"Resolution", resolutionString},
            }
        },
            result => Debug.Log("Successfully updated user data"),
            error =>
            {
                Debug.Log("Got error setting user data Ancestor to Arthur");
                Debug.Log(error.GenerateErrorReport());
            });
    }

    public void ResetPrefs()
    {
        PlayerPrefs.DeleteAll();
    }   
}


