(function() {
  const fn = function() {
    (function(root) {
      function now() {
        return new Date();
      }
    
      const force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
    
    const element = document.getElementById("6cf58c52-b94b-456f-b9bb-808f801971a7");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '6cf58c52-b94b-456f-b9bb-808f801971a7' but no matching script tag was found.")
        }
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error(url) {
          console.error("failed to load " + url);
        }
    
        for (let i = 0; i < css_urls.length; i++) {
          const url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error.bind(null, url);
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        for (let i = 0; i < js_urls.length; i++) {
          const url = js_urls[i];
          const element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error.bind(null, url);
          element.async = false;
          element.src = url;
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      const js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-gl-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.0.2.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-mathjax-3.0.2.min.js"];
      const css_urls = [];
    
      const inline_js = [    function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        function(Bokeh) {
          (function() {
            const fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                  const docs_json = '{"6edd1b25-cb48-4c1e-9e50-589a01daee29":{"version":"3.0.2","title":"Bokeh Application","defs":[],"roots":[{"type":"object","name":"GridPlot","id":"p9189","attributes":{"toolbar":{"type":"object","name":"Toolbar","id":"p9188","attributes":{"tools":[{"type":"object","name":"ResetTool","id":"p9125"},{"type":"object","name":"PanTool","id":"p9126"},{"type":"object","name":"BoxZoomTool","id":"p9127","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p9128","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"WheelZoomTool","id":"p9129"},{"type":"object","name":"LassoSelectTool","id":"p9130","attributes":{"renderers":"auto","overlay":{"type":"object","name":"PolyAnnotation","id":"p9131","attributes":{"syncable":false,"level":"overlay","visible":false,"xs":[],"xs_units":"canvas","ys":[],"ys_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"UndoTool","id":"p9132"},{"type":"object","name":"SaveTool","id":"p9187"},{"type":"object","name":"HoverTool","id":"p9134","attributes":{"renderers":"auto"}}]}},"children":[[{"type":"object","name":"Figure","id":"p9094","attributes":{"width":500,"height":500,"x_range":{"type":"object","name":"DataRange1d","id":"p9095"},"y_range":{"type":"object","name":"DataRange1d","id":"p9096"},"x_scale":{"type":"object","name":"LinearScale","id":"p9107"},"y_scale":{"type":"object","name":"LinearScale","id":"p9109"},"title":{"type":"object","name":"Title","id":"p9098"},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p9152","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p9146","attributes":{"selected":{"type":"object","name":"Selection","id":"p9147","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p9148"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"Q9HErslqB8BD0cSuyWoHwHXIOcde+AbAp7+u3/OFBsDZtiP4iBMGwAyumBAeoQXAPqUNKbMuBcBwnIJBSLwEwKKT91ndSQTA1IpscnLXA8AGguGKB2UDwDh5VqOc8gLAanDLuzGAAsCdZ0DUxg0CwM9etexbmwHAAVYqBfEoAcAzTZ8dhrYAwGVEFDYbRADAL3cSnWCj/7+TZfzNir7+v/dT5v602f2/XELQL9/0/L/AMLpgCRD8vyQfpJEzK/u/iA2Owl1G+r/t+3fzh2H5v1HqYSSyfPi/tdhLVdyX978axzWGBrP2v361H7cwzvW/4qMJ6Frp9L9GkvMYhQT0v6uA3UmvH/O/D2/Hetk68r9zXbGrA1bxv9hLm9wtcfC/eHQKG7AY779AUd58BE/tvwgust5Yheu/0AqGQK276b+Y51miAfLnv2TELQRWKOa/LKEBZqpe5L/0fdXH/pTiv7xaqSlTy+C/CG/6Fk8D3r+YKKLa92/avyjiSZ6g3Na/wJvxYUlJ07+gqjJL5GvPv8AdgtI1Rci/4JDRWYcewb8ACELCse+zvwC5g0NTiZa/AFcAQRBWoT8gReERZfi2P3AvoQHhosI/ULxReo/JyT+YJIH5HnjQPwhr2TV2C9Q/eLExcs2e1z/o94muJDLbP1A+4up7xd4/YEKdk2ks4T+YZckxFfbiP9CI9c/Av+Q/CKwhbmyJ5j9Az00MGFPoP3jyearDHOo/rBWmSG/m6z/kONLmGrDtPxxc/oTGee8/qj+VEbmh8D9GUavgjobxP+Jiwa9ka/I/fnTXfjpQ8z8ahu1NEDX0P7aXAx3mGfU/UqkZ7Lv+9T/uui+7keP2P4bMRYpnyPc/It5bWT2t+D++73EoE5L5P1oBiPfodvo/9hKexr5b+z+SJLSVlED8Py42ymRqJf0/ykfgM0AK/j9mWfYCFu/+PwJrDNLr0/8/Tz6R0GBcAEAdRxy4y84AQOtPp582QQFAuVgyh6GzAUCFYb1uDCYCQFNqSFZ3mAJAIXPTPeIKA0Dve14lTX0DQL2E6Qy47wNAi4109CJiBEBYlv/bjdQEQA=="},"shape":[101],"dtype":"float64","order":"little"}],["y",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAAAAAD8qfHSTWJQP/yp8dJNYlA//Knx0k1iUD/6fmq8dJNoP/yp8dJNYnA//Knx0k1icD/6fmq8dJN4P3npJjEIrHw/O99PjZdugj+6SQwCK4eGP7gehetRuI4/nMQgsHJokT8730+Nl26SP9v5fmq8dJM/exSuR+F6lD9aZDvfT42XP5qZmZmZmZk/eekmMQisnD+4HoXrUbieP+xRuB6F66E/exSuR+F6pD8bL90kBoGlP0oMAiuHFqk/2c73U+Olqz9YObTIdr6vP0SLbOf7qbE/g8DKoUW2sz/D9Shcj8K1P7Kd76fGS7c/6SYxCKwcuj/RItv5fmq8P7gehetRuL4/+FPjpZvEwD/n+6nx0k3CPy/dJAaBlcM/Gy/dJAaBxT9aZDvfT43HP05iEFg5tMg/ke18PzVeyj/FILByaJHNP6wcWmQ7388/SgwCK4cW0T+TGARWDi3SPy/dJAaBldM/0SLb+X5q1D/FILByaJHVP2Dl0CLb+dY/ppvEILBy2D/n+6nx0k3aP9v5fmq8dNs/dZMYBFYO3T8QWDm0yHbePwIrhxbZzt8/pHA9Ctej4D8fhetRuB7hP+58PzVeuuE/j8L1KFyP4j+0yHa+nxrjP9V46SYxCOQ/eekmMQis5D9I4XoUrkflPxkEVg4tsuU/aJHtfD815j+28/3UeOnmP4cW2c73U+c/f2q8dJMY6D+kcD0K16PoPx1aZDvfT+k/7FG4HoXr6T+R7Xw/NV7qP42XbhKDwOo/MzMzMzMz6z8xCKwcWmTrPwIrhxbZzus/VOOlm8Qg7D97FK5H4XrsP83MzMzMzOw/TDeJQWDl7D9zaJHtfD/tP3E9CtejcO0/RIts5/up7T8X2c73U+PtP2q8dJMYBO4/5/up8dJN7j/l0CLb+X7uPw4tsp3vp+4/tvP91Hjp7j/fT42XbhLvP7Kd76fGS+8/MQisHFpk7z+wcmiR7XzvP1pkO99Pje8/WmQ730+N7z8EVg4tsp3vP4PAyqFFtu8/WDm0yHa+7z8CK4cW2c7vP4GVQ4ts5+8/Vg4tsp3v7z8AAAAAAADwPw=="},"shape":[101],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p9153","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p9154"}}},"glyph":{"type":"object","name":"Step","id":"p9149","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#1f77b4","mode":"after"}},"nonselection_glyph":{"type":"object","name":"Step","id":"p9150","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#1f77b4","line_alpha":0.1,"mode":"after"}},"muted_glyph":{"type":"object","name":"Step","id":"p9151","attributes":{"x":{"type":"field","field":"x"},"y":{"type":"field","field":"y"},"line_color":"#1f77b4","line_alpha":0.2,"mode":"after"}}}},{"type":"object","name":"GlyphRenderer","id":"p9161","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p9155","attributes":{"selected":{"type":"object","name":"Selection","id":"p9156","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p9157"},"data":{"type":"map","entries":[["x",{"type":"ndarray","array":{"type":"bytes","data":"Q9HErslqB8B1yDnHXvgGwKe/rt/zhQbA2bYj+IgTBsAMrpgQHqEFwD6lDSmzLgXAcJyCQUi8BMCik/dZ3UkEwNSKbHJy1wPABoLhigdlA8A4eVajnPICwGpwy7sxgALAnWdA1MYNAsDPXrXsW5sBwAFWKgXxKAHAM02fHYa2AMBlRBQ2G0QAwC93Ep1go/+/k2X8zYq+/r/3U+b+tNn9v1xC0C/f9Py/wDC6YAkQ/L8kH6SRMyv7v4gNjsJdRvq/7ft384dh+b9R6mEksnz4v7XYS1Xcl/e/Gsc1hgaz9r9+tR+3MM71v+KjCeha6fS/RpLzGIUE9L+rgN1Jrx/zvw9vx3rZOvK/c12xqwNW8b/YS5vcLXHwv3h0ChuwGO+/QFHefARP7b8ILrLeWIXrv9AKhkCtu+m/mOdZogHy579kxC0EVijmvyyhAWaqXuS/9H3Vx/6U4r+8WqkpU8vgvwhv+hZPA96/mCii2vdv2r8o4kmeoNzWv8Cb8WFJSdO/oKoyS+Rrz7/AHYLSNUXIv+CQ0VmHHsG/AAhCwrHvs78AuYNDU4mWvwBXAEEQVqE/IEXhEWX4tj9wL6EB4aLCP1C8UXqPyck/mCSB+R540D8Ia9k1dgvUP3ixMXLNntc/6PeJriQy2z9QPuLqe8XeP2BCnZNpLOE/mGXJMRX24j/QiPXPwL/kPwisIW5sieY/QM9NDBhT6D948nmqwxzqP6wVpkhv5us/5DjS5hqw7T8cXP6ExnnvP6o/lRG5ofA/RlGr4I6G8T/iYsGvZGvyP3501346UPM/GobtTRA19D+2lwMd5hn1P1KpGey7/vU/7rovu5Hj9j+GzEWKZ8j3PyLeW1k9rfg/vu9xKBOS+T9aAYj36Hb6P/YSnsa+W/s/kiS0lZRA/D8uNspkaiX9P8pH4DNACv4/Zln2Ahbv/j8CawzS69P/P08+kdBgXABAHUccuMvOAEDrT6efNkEBQLlYMoehswFAhWG9bgwmAkBTakhWd5gCQCFz0z3iCgNA73teJU19A0C9hOkMuO8DQIuNdPQiYgRAWJb/243UBEA="},"shape":[100],"dtype":"float64","order":"little"}],["y1",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/Knx0k1iUD/8qfHSTWJQP/yp8dJNYmA/+n5qvHSTaD/8qfHSTWJwP3sUrkfhenQ/+n5qvHSTeD/8qfHSTWKAP3sUrkfheoQ/+n5qvHSTiD+4HoXrUbiOPzvfT42XbpI/ukkMAiuHlj+amZmZmZmZP7gehetRuJ4/7FG4HoXroT97FK5H4XqkP1pkO99Pjac/ObTIdr6fqj9oke18PzWuP/T91HjpJrE/2/l+arx0sz/D9Shcj8K1P1K4HoXrUbg/iUFg5dAiuz/ByqFFtvO9P1CNl24Sg8A/kxgEVg4twj8rhxbZzvfDPxfZzvdT48U/AiuHFtnOxz9CYOXQItvJP9V46SYxCMw/aJHtfD81zj+oxks3iUHQP0a28/3UeNE/jZduEoPA0j/VeOkmMQjUPx1aZDvfT9U/Di2yne+n1j8AAAAAAADYP5zEILByaNk/N4lBYOXQ2j/TTWIQWDncP28Sg8DKod0/CtejcD0K3z+oxks3iUHgP/YoXI/C9eA/RIts5/up4T+R7Xw/NV7iPwrXo3A9CuM/WDm0yHa+4z/8qfHSTWLkP3WTGARWDuU/GQRWDi2y5T/n+6nx0k3mP7bz/dR46eY/sHJoke185z/VeOkmMQjoP/p+arx0k+g/SgwCK4cW6T/FILByaJHpPz81XrpJDOo/5dAi2/l+6j+28/3UeOnqP7Kd76fGS+s/rkfhehSu6z/VeOkmMQjsPycxCKwcWuw/eekmMQis7D/2KFyPwvXsP57vp8ZLN+0/Rrbz/dR47T8ZBFYOLbLtPxfZzvdT4+0/6SYxCKwc7j8Sg8DKoUXuPzvfT42Xbu4/ZDvfT42X7j+4HoXrUbjuPwwCK4cW2e4/i2zn+6nx7j/fT42XbhLvP166SQwCK+8/CKwcWmQ77z+HFtnO91PvPzEIrBxaZO8/2/l+arx07z+wcmiR7XzvP1pkO99Pje8/L90kBoGV7z8="},"shape":[100],"dtype":"float64","order":"little"}],["y2",{"type":"ndarray","array":{"type":"bytes","data":"eekmMQisfD/8qfHSTWKAPzvfT42XboI/exSuR+F6hD+6SQwCK4eGP/p+arx0k4g/ObTIdr6fij956SYxCKyMP/yp8dJNYpA/O99PjZdukj97FK5H4XqUP7pJDAIrh5Y/+n5qvHSTmD/ZzvdT46WbP7gehetRuJ4/TDeJQWDloD+LbOf7qfGiP8uhRbbz/aQ/CtejcD0Kpz+amZmZmZmpPylcj8L1KKw/CKwcWmQ7rz/0/dR46SaxP4ts5/up8bI/I9v5fmq8tD9iEFg5tMi2P6JFtvP91Lg/iUFg5dAiuz8ZBFYOLbK9P1TjpZvEIMA/RIts5/upwT/fT42XbhLDPyPb+X5qvMQ/ZmZmZmZmxj/+1HjpJjHIP+kmMQisHMo/1XjpJjEIzD9oke18PzXOP/7UeOkmMdA/SOF6FK5H0T8730+Nl27SP9nO91PjpdM/IbByaJHt1D9oke18PzXWP7ByaJHtfNc/okW28/3U2D+TGARWDi3aPy/dJAaBlds/y6FFtvP93D9mZmZmZmbePwIrhxbZzt8/z/dT46Wb4D/y0k1iEFjhPz81XrpJDOI/jZduEoPA4j/b+X5qvHTjP1TjpZvEIOQ/zczMzMzM5D9GtvP91HjlP76fGi/dJOY/YhBYObTI5j8xCKwcWmTnPwAAAAAAAOg/+n5qvHST6D8fhetRuB7pP0SLbOf7qek/vp8aL90k6j8OLbKd76fqP7TIdr6fGus/WmQ730+N6z9WDi2yne/rP1K4HoXrUew/TmIQWDm07D+gGi/dJAbtP/LSTWIQWO0/bxKDwMqh7T8X2c73U+PtP76fGi/dJO4/ke18PzVe7j+PwvUoXI/uP42XbhKDwO4/i2zn+6nx7j/fT42XbhLvPzMzMzMzM+8/hxbZzvdT7z/b+X5qvHTvP4XrUbgehe8/BFYOLbKd7z+uR+F6FK7vP1g5tMh2vu8/AiuHFtnO7z/Xo3A9CtfvP6wcWmQ73+8/gZVDi2zn7z9WDi2yne/vPyuHFtnO9+8/K4cW2c737z8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8="},"shape":[100],"dtype":"float64","order":"little"}]]}}},"view":{"type":"object","name":"CDSView","id":"p9162","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p9163"}}},"glyph":{"type":"object","name":"VArea","id":"p9158","attributes":{"x":{"type":"field","field":"x"},"y1":{"type":"field","field":"y1"},"y2":{"type":"field","field":"y2"},"fill_color":"#107591","fill_alpha":0.2}},"nonselection_glyph":{"type":"object","name":"VArea","id":"p9159","attributes":{"x":{"type":"field","field":"x"},"y1":{"type":"field","field":"y1"},"y2":{"type":"field","field":"y2"},"fill_color":"#107591","fill_alpha":0.1,"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"VArea","id":"p9160","attributes":{"x":{"type":"field","field":"x"},"y1":{"type":"field","field":"y1"},"y2":{"type":"field","field":"y2"},"fill_color":"#107591","fill_alpha":0.2,"hatch_alpha":{"type":"value","value":0.2}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p9100","attributes":{"tools":[{"id":"p9125"},{"id":"p9126"},{"id":"p9127"},{"id":"p9129"},{"id":"p9130"},{"id":"p9132"},{"type":"object","name":"SaveTool","id":"p9133"},{"id":"p9134"}]}},"toolbar_location":null,"left":[{"type":"object","name":"LinearAxis","id":"p9118","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p9121","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p9120"},"major_label_policy":{"type":"object","name":"AllLabels","id":"p9119"}}}],"below":[{"type":"object","name":"LinearAxis","id":"p9111","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"p9114","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"p9113"},"major_label_policy":{"type":"object","name":"AllLabels","id":"p9112"}}}],"center":[{"type":"object","name":"Grid","id":"p9117","attributes":{"axis":{"id":"p9111"}}},{"type":"object","name":"Grid","id":"p9124","attributes":{"dimension":1,"axis":{"id":"p9118"}}}],"output_backend":"webgl"}},0,0]]}}]}}';
                  const render_items = [{"docid":"6edd1b25-cb48-4c1e-9e50-589a01daee29","roots":{"p9189":"6cf58c52-b94b-456f-b9bb-808f801971a7"},"root_ids":["p9189"]}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    let attempts = 0;
                    const timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
    function(Bokeh) {
        }
      ];
    
      function run_inline_js() {
        for (let i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();