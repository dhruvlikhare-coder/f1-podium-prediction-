/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        f1: {
          red: '#E10600',
          scriptRed: '#D91424',
          champagne: '#E2C8A2',
          gold: '#D4AF37',
          dark: '#050608',
          card: '#0D0F17',
          border: 'rgba(255, 255, 255, 0.08)',
          cyan: '#00F5D4',
        }
      },
      fontFamily: {
        cursive: ['"Great Vibes"', '"Alex Brush"', 'cursive'],
        display: ['Montserrat', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
        sans: ['"Space Grotesk"', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        }
      }
    },
  },
  plugins: [],
}
