const { app, Tray, Menu, nativeImage, shell, BrowserWindow, Notification } = require('electron');
const screenshot = require('screenshot-desktop');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { runAutomationCommand } = require('./dist/computerUseGraph.js');

let tray = null;

// Create a simple icon for the tray
function createTrayIcon() {
  // Create a simple emoji-based icon - easier and more visible
  const icon = nativeImage.createFromNamedImage('NSImageNameComputer', [16, 16]);
  if (icon.isEmpty()) {
    // Fallback: create from text
    const canvas = { width: 16, height: 16 };
    const buffer = Buffer.alloc(canvas.width * canvas.height * 4); // RGBA
    
    // Fill with black pixels to make a visible square
    for (let i = 0; i < buffer.length; i += 4) {
      buffer[i] = 0;     // R
      buffer[i + 1] = 0; // G  
      buffer[i + 2] = 0; // B
      buffer[i + 3] = 255; // A (fully opaque)
    }
    
    return nativeImage.createFromBuffer(buffer, canvas);
  }
  console.log('Created tray icon:', icon.getSize());
  return icon;
}

function takeScreenshot() {
  console.log('Taking screenshot...');
  
  // Debug: Try different screenshot methods
  screenshot.all().then((imgData) => {
    console.log('Screenshot data type:', typeof imgData);
    console.log('Is Array?', Array.isArray(imgData));
    console.log('Is Buffer?', Buffer.isBuffer(imgData));
    console.log('Data length/size:', imgData?.length || 'undefined');
    console.log('First few bytes:', imgData?.slice ? imgData.slice(0, 10) : 'no slice method');
    
    if (!imgData || (Array.isArray(imgData) && imgData.length === 0) || (Buffer.isBuffer(imgData) && imgData.length === 0)) {
      console.error('Screenshot data is empty or invalid');
      
      // Try fallback method using native macOS screencapture
      const { exec } = require('child_process');
      const desktopPath = path.join(os.homedir(), 'Desktop');
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `screenshot-${timestamp}.png`;
      const filepath = path.join(desktopPath, filename);
      
      console.log('Trying native macOS screencapture...');
      exec(`screencapture "${filepath}"`, (error, stdout, stderr) => {
        if (error) {
          console.error('Error with screencapture:', error);
        } else {
          console.log(`Screenshot saved via screencapture to: ${filepath}`);
          shell.showItemInFolder(filepath);
        }
      });
      return;
    }
    
    // Convert to Buffer if it's an Array
    let imgBuffer;
    if (Array.isArray(imgData)) {
      imgBuffer = Buffer.from(imgData);
    } else if (Buffer.isBuffer(imgData)) {
      imgBuffer = imgData;
    } else {
      imgBuffer = Buffer.from(imgData);
    }
    
    // Get desktop path
    const desktopPath = path.join(os.homedir(), 'Desktop');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `screenshot-${timestamp}.png`;
    const filepath = path.join(desktopPath, filename);
    
    console.log(`Saving screenshot to: ${filepath}`);
    console.log(`Buffer size: ${imgBuffer.length} bytes`);
    
    // Save the screenshot
    fs.writeFile(filepath, imgBuffer, (err) => {
      if (err) {
        console.error('Error saving screenshot:', err);
      } else {
        console.log(`Screenshot saved successfully to: ${filepath}`);
        // Show the file in Finder
        shell.showItemInFolder(filepath);
      }
    });
  }).catch((err) => {
    console.error('Error taking screenshot:', err);
    
    // Fallback to native macOS screencapture
    const { exec } = require('child_process');
    const desktopPath = path.join(os.homedir(), 'Desktop');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `screenshot-${timestamp}.png`;
    const filepath = path.join(desktopPath, filename);
    
    console.log('Falling back to native macOS screencapture...');
    exec(`screencapture "${filepath}"`, (error, stdout, stderr) => {
      if (error) {
        console.error('Error with screencapture:', error);
      } else {
        console.log(`Screenshot saved via screencapture to: ${filepath}`);
        shell.showItemInFolder(filepath);
      }
    });
  });
}

// Function to show automation result to user
function showAutomationResult(command, result, isSuccess = true) {
  // Show notification
  const notification = new Notification({
    title: isSuccess ? '✅ Automation Complete' : '❌ Automation Failed',
    body: isSuccess ? `Successfully completed: ${command}` : `Failed to complete: ${command}`,
    silent: false
  });
  
  notification.show();
  }

// Function to show input dialog for automation command
async function showCommandInputDialog() {
  return new Promise((resolve) => {
    const inputWindow = new BrowserWindow({
      width: 500,
      height: 500,
      show: false,
      resizable: false,
      minimizable: false,
      maximizable: false,
      alwaysOnTop: true,
      webPreferences: {
        nodeIntegration: true,
        contextIsolation: false
      }
    });

    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>Automation Command</title>
        <style>
          body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
            padding: 20px; 
            margin: 0;
            background: #f5f5f5;
          }
          .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
          }
          h2 { margin-top: 0; color: #333; }
          input { 
            width: 100%; 
            padding: 10px; 
            font-size: 16px; 
            border: 1px solid #ddd; 
            border-radius: 4px; 
            margin: 10px 0;
            box-sizing: border-box;
          }
          .buttons { 
            text-align: right; 
            margin-top: 15px; 
          }
          button { 
            padding: 8px 16px; 
            margin-left: 10px; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer;
            font-size: 14px;
          }
          .run-btn { 
            background: #007AFF; 
            color: white; 
          }
          .cancel-btn { 
            background: #f0f0f0; 
            color: #333; 
          }
          button:hover { opacity: 0.8; }
        </style>
      </head>
      <body>
        <div class="container">
          <h2>🤖 Automation Command</h2>
          <p>Enter the action you want the AI to perform:</p>
          <input type="text" id="commandInput" placeholder="e.g., click on Finder, open Safari, type hello world" autofocus>
          <div class="buttons">
            <button class="cancel-btn" onclick="cancel()">Cancel</button>
            <button class="run-btn" onclick="runCommand()">Run Automation</button>
          </div>
        </div>
        
        <script>
          const { ipcRenderer } = require('electron');
          
          function runCommand() {
            const command = document.getElementById('commandInput').value.trim();
            if (command) {
              ipcRenderer.send('automation-command', command);
            }
          }
          
          function cancel() {
            ipcRenderer.send('automation-command', null);
          }
          
          // Handle Enter key
          document.getElementById('commandInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
              runCommand();
            }
          });
          
          // Handle Escape key
          document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
              cancel();
            }
          });
        </script>
      </body>
      </html>
    `;

    inputWindow.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(htmlContent));
    
    inputWindow.once('ready-to-show', () => {
      inputWindow.show();
      inputWindow.focus();
    });

    inputWindow.webContents.once('ipc-message', (event, channel, command) => {
      if (channel === 'automation-command') {
        inputWindow.close();
        resolve(command);
      }
    });

    inputWindow.on('closed', () => {
      resolve(null);
    });
  });
}

async function runAutomation() {
  console.log('🤖 Getting automation command from user...');
  
  try {
    const command = await showCommandInputDialog();
    
    if (!command) {
      console.log('Automation cancelled by user');
      return;
    }
    
    console.log(`🤖 Running automation: ${command}...`);
    const result = await runAutomationCommand(command);
    console.log('Automation completed:', result);
    
    // Show visual cue for successful completion
    showAutomationResult(command, result, true);
  } catch (error) {
    console.error('Error running automation:', error);
    console.error('Make sure OPENAI_API_KEY is set in your environment');
    
    // Show visual cue for failed automation
    const errorMessage = error.message || 'Unknown error occurred. Make sure OPENAI_API_KEY is set in your environment.';
    showAutomationResult(command, errorMessage, false);
  }
}

function createTray() {
  console.log('Creating tray...');
  const icon = createTrayIcon();
  console.log('Icon created, empty?', icon.isEmpty());
  
  tray = new Tray(icon);
  console.log('Tray created successfully');
  
  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Take Screenshot',
      click: takeScreenshot
    },
    {
      type: 'separator'
    },
    {
      label: 'Automate',
      click: runAutomation
    },
    {
      type: 'separator'
    },
    {
      label: 'Quit',
      click: () => {
        app.quit();
      }
    }
  ]);
  
  tray.setToolTip('Screenshot App');
  tray.setContextMenu(contextMenu);
  console.log('Tray setup complete');
}

// This method will be called when Electron has finished initialization
app.whenReady().then(() => {
  createTray();
  
  // Hide the app from the dock on macOS
  if (process.platform === 'darwin') {
    app.dock.hide();
  }
});

// Quit when all windows are closed
app.on('window-all-closed', () => {
  // On macOS, keep the app running even when all windows are closed
  // since we're running as a tray app
});

app.on('activate', () => {
  // On macOS, re-create the tray if it doesn't exist
  if (tray === null) {
    createTray();
  }
});

// Prevent the app from quitting when the last window is closed
app.on('before-quit', () => {
  if (tray) {
    tray.destroy();
  }
}); 