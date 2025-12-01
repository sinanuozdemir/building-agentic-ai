# Screenshot Tray App

A simple Electron app that runs in your system tray and allows you to take screenshots with a single click.

## Features

- Runs quietly in your system tray
- Right-click the tray icon to see options
- **"Take Screenshot"** captures your entire screen
- **"Automate"** uses AI to click on Finder (with LangGraph + OpenAI)
- Screenshots are automatically saved to your Desktop with timestamps
- Automatically opens Finder to show the saved screenshot

## Installation & Setup

1. Make sure you have Node.js installed on your Mac
2. Navigate to this directory in Terminal:
   ```bash
   cd screenshot_app
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Set up OpenAI API key for automation features:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

5. Run the app:
   ```bash
   npm start
   ```

## Usage

1. After running `npm start`, you'll see a small icon appear in your system tray (top-right of your screen)
2. Right-click on the tray icon
3. Select "Take Screenshot" from the menu
4. Your screenshot will be saved to your Desktop and Finder will open to show the file
5. To quit the app, right-click the tray icon and select "Quit"

## Notes

- The app is designed for macOS (your M1 Mac)
- Screenshots are saved as PNG files with timestamps
- The app runs in the background and doesn't show in your Dock
- You may need to grant screen recording permissions when first running the app

## Troubleshooting

If you get permission errors:
1. Go to System Preferences > Security & Privacy > Privacy
2. Select "Screen Recording" from the left sidebar
3. Check the box next to "Electron" or your app name
4. Restart the app 