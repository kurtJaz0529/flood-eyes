; 慧眼识灾 · Windows 安装包脚本（Inno Setup 6）
; ===============================================
; 特点：
;   · 单文件 setup.exe，双击即装，无需管理员权限（装到用户目录）
;   · 自动创建开始菜单 + 桌面快捷方式，注册"添加/删除程序"卸载项
;   · 安装目录可写 -> 识别结果与 PDF 简报就存在程序目录的 outputs 文件夹
;   · 安装完可选择立即启动
;
; 编译（两种版本）：
;   ISCC.exe /DProfile=lite build\installer.iss    -> 慧眼识灾_安装程序_v0.3.1_精简版.exe
;   ISCC.exe /DProfile=full build\installer.iss    -> 慧眼识灾_安装程序_v0.3.1_完整版.exe
;
; 也可以用 build/build_installer.ps1 一键编译。

#ifndef Profile
  #define Profile "lite"
#endif

#if Profile == "full"
  #define SourceDir "..\dist_full\慧眼识灾"
  #define EditionName "完整版"
  #define EditionNote "含 PyTorch + U-Net 深度模型"
#else
  #define SourceDir "..\dist\慧眼识灾"
  #define EditionName "精简版"
  #define EditionNote "NDWI + Otsu 基线（启动快、体积小）"
#endif

#define AppName "慧眼识灾"
#define AppVersion "0.3.1"
#define AppPublisher "慧眼识灾团队"
#define AppExeName "慧眼识灾.exe"

[Setup]
; AppId 固定，保证升级时能覆盖安装而不是并存两份
AppId={{8F3C2A41-6D2B-4E7A-9C15-1B7E4F0A2D33}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} V{#AppVersion}（{#EditionName}）
AppPublisher={#AppPublisher}
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} 安装程序（{#EditionName}）
VersionInfoTextVersion={#AppVersion}
AppMutex=HuiYanShiZai_SingleInstance_Mutex
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=no
AllowNoIcons=yes
; 用户级安装：不弹 UAC，普通账号也能装，装到 %LOCALAPPDATA%\Programs
PrivilegesRequired=lowest
OutputDir=..\dist_installer
OutputBaseFilename={#AppName}_安装程序_v{#AppVersion}_{#EditionName}
SetupIconFile=..\docs\assets\app_icon.ico
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} V{#AppVersion}（{#EditionName}）
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
MinVersion=10.0
ShowLanguageDialog=no
DisableWelcomePage=no
AppComments={#EditionNote}

[Languages]
Name: "chinese"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务："; Flags: checkedonce

[Files]
; 主程序（整个 _internal 目录一并装进去）
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
; 提前建好输出目录，用户装完就能看到结果放在哪
Name: "{app}\outputs"
Name: "{app}\logs"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Comment: "启动 {#AppName}"
Name: "{group}\使用说明"; Filename: "{app}\使用说明.txt"
Name: "{group}\识别结果目录"; Filename: "{app}\outputs"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "立即启动 {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 卸载时清掉运行产生的数据
Type: filesandordirs; Name: "{app}\outputs"
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\_internal"
