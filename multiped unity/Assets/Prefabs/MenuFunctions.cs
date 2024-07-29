using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class MenuFunctions : MonoBehaviour
{
    public GameObject loading;


    // Add scenes to build settings
    // Change scene name for StartScene
    public void StartTrialScene()
    {
        loading.SetActive(true);
        StartCoroutine(SceneStarter());
        
    }

    private IEnumerator SceneStarter()
    {
        AsyncOperation asyncLoad = SceneManager.LoadSceneAsync("Environment");

        // Wait until the asynchronous scene fully loads
        while (!asyncLoad.isDone)
        {
            yield return null;
        }
 
    }

    public void Quit ()
    {
        Debug.Log("Quit");
        Application.Quit();
    }


}
