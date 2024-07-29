using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class menuscript : MonoBehaviour
{
    public void StartBtn()
    {
        SceneManager.LoadScene("Environment");
    }
    public void ExitBtn()
    {
        Application.Quit();
    }
}
