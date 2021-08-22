SET DISTUTILS_USE_SDK=1
SET env_path_root=C:\Program Files (x86)\Microsoft Visual Studio\2019
SET env_path_head=VC\Auxiliary\Build\vcvars64.bat

echo off
IF EXIST "%env_path_root%\BuildTools\%env_path_head%" (
    "%env_path_root%\BuildTools\%env_path_head%"
) ELSE (
    IF EXIST "%build_root_path%\Community\%env_path_head%" (
        "%build_root_path%\Community\%env_path_head%"
    ) ELSE (
        IF EXIST "%build_root_path%\Professional\%env_path_head%" (
            "%build_root_path%\Professional\%env_path_head%"
        ) ELSE (
            IF EXIST "%build_root_path%\Entreprise\%env_path_head%" (
                "%build_root_path%\Entreprise\%env_path_head%"
            ) ELSE (
                echo Visual Studio 2019 required.
            )
        )
    )
)
