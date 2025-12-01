import path from 'path';
import { fileURLToPath } from 'url';

import HtmlWebpackPlugin from 'html-webpack-plugin';
import CopyPlugin from 'copy-webpack-plugin';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const config = {
    mode: 'development',
    devtool: 'inline-source-map',
    entry: {
        background: {
            import: './src/background.js',
            chunkLoading: `import-scripts`,
        },
        popup: './src/popup.js',
        content: './src/content.js',
        'inject-classifier': './src/inject-classifier.js',
    },
    output: {
        path: path.resolve(__dirname, 'build'),
        filename: '[name].js',
    },
    resolve: {
        fallback: {
            fs: false,
            path: false,
            url: false,
            crypto: false
        }
    },
    plugins: [
        new HtmlWebpackPlugin({
            template: './src/popup.html',
            filename: 'popup.html',
        }),
        new CopyPlugin({
            patterns: [
                {
                    from: "public",
                    to: "." // Copies to build folder
                },
                {
                    from: "src/popup.css",
                    to: "popup.css"
                },
                // Copy all transformers.js WASM files
                {
                    from: "node_modules/@huggingface/transformers/dist/*.wasm",
                    to: "[name][ext]"
                },
                {
                    from: "node_modules/@huggingface/transformers/dist/*.jsep.mjs",
                    to: "[name][ext]"
                }
            ],
        })
    ],
};

export default config;
