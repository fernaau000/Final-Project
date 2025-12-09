const express = require('express');
const path = require('path');
const fs = require('fs');
const app = express();
const PORT = process.env.PORT || 8000;

const distDir = path.join(__dirname, 'dist');
if (fs.existsSync(distDir)) {
  // Serve optimized build when `dist/` exists
  app.use(express.static(distDir));
  app.get('*', (req, res) => {
    res.sendFile(path.join(distDir, 'index.html'));
  });
  console.log('Serving `dist/` (production build)');
} else {
  // Fallback: serve project root (dev)
  app.use(express.static(path.join(__dirname)));
  app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
  });
  console.log('Serving project root (dev)');
}

app.listen(PORT, () => {
  console.log(`Static server running at http://localhost:${PORT}`);
});
